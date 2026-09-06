"""Script orchestration checks with optional ML/simulator dependencies replaced.

These tests exercise data flow and lifecycle behavior, not model numerics.
Run with: python3 -m pytest -q tests/test_scripts.py
"""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def load_script(monkeypatch):
    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    serialization = stub('flax.serialization', to_bytes=Mock(return_value=b'params'))
    stub('flax', serialization=serialization)
    stub('jax', numpy=np, random=SimpleNamespace(key=lambda seed: seed))
    monkeypatch.setitem(sys.modules, 'jax.numpy', np)
    stub('torch', cuda=SimpleNamespace(empty_cache=Mock()))
    stub('tyro', cli=Mock())
    stub('scipy.spatial.transform', Rotation=Mock())
    stub('src.my_vla.data.libero', DEFAULT_LIBERO_DATA_ROOT=Path('/data'),
         LiberoConfig=lambda root, horizon: (root, horizon), iter_libero_transitions=Mock())
    stub('src.my_vla.models.base_vlm', GrootN15Adapter=Mock())
    stub('src.my_vla.models.future_state', ActionConditionedTransition=Mock(), future_consistency=Mock())
    stub('src.my_vla.models.projector', LatentProjector=Mock())
    stub('src.my_vla.rl.residual_ac', ResidualActor=Mock(), build_actor_input=Mock(), clip_libero_action=Mock())
    stub('src.my_vla.training.pretrain', PretrainConfig=Mock(), initialize_pretraining=Mock(), pretrain_step=Mock())
    stub('src.my_vla.models.residual_libero_rollout', ResidualLiberoRolloutPolicy=Mock())
    stub('openpi.serving', websocket_policy_server=SimpleNamespace(WebsocketPolicyServer=Mock()))
    stub('imageio', mimwrite=Mock())
    stub('tqdm', tqdm=lambda values: values)
    stub('libero.libero', benchmark=Mock(), get_libero_path=Mock())
    stub('libero.libero.envs', OffScreenRenderEnv=Mock())
    stub('openpi_client', image_tools=Mock(), websocket_client_policy=Mock())

    def load(relative):
        name = '_script_test_' + Path(relative).stem
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    bank = load('src/my_vla/retrieval/bank.py')
    monkeypatch.setitem(sys.modules, 'src.my_vla.retrieval.bank', bank)
    return load


def test_training_features_and_batch_preserve_checkpoint_tail(load_script, monkeypatch):
    module = load_script('scripts/train_residual_libero.py')
    sample = {
        'observation': {}, 'instruction': 'move', 'state': np.zeros(7),
        'state_chunk': np.arange(35).reshape(5, 7),
        'expert_action_chunk': np.ones((4, 7)), 'return_to_go': 1,
    }
    monkeypatch.setattr(module, 'iter_libero_transitions', lambda config: iter([sample, sample]))
    adapter = Mock(return_value=SimpleNamespace(base_action=np.zeros((2, 7)), hidden=np.ones((2, 3))))
    config = module.TrainConfig('model', horizon=4, replan_steps=2, max_samples=1)
    records, samples = module.LiberoSampleCollector(config, adapter).collect()
    assert len(samples) == 1
    np.testing.assert_array_equal(records[0].state, sample['state_chunk'][2])
    np.testing.assert_array_equal(records[0].residual_target, np.ones(14))
    bank = module.RetrievalBank(records=records)
    batch = module.ResidualLiberoTrainer(config)._make_batch(samples, records, bank)
    assert batch['base_action_prefix'].shape == (1, 2, 7)
    assert batch['base_action_tail'].shape == (1, 2, 7)
    np.testing.assert_array_equal(batch['future_state'][0], sample['state_chunk'][2])
    np.testing.assert_array_equal(batch['residual_target'], np.ones((1, 2, 7)))


def test_training_empty_dataset_and_invalid_batch(load_script, monkeypatch):
    module = load_script('scripts/train_residual_libero.py')
    monkeypatch.setattr(module, 'iter_libero_transitions', lambda config: iter([]))
    with pytest.raises(RuntimeError, match='No valid LIBERO'):
        module.LiberoSampleCollector(module.TrainConfig('model'), Mock()).collect()
    with pytest.raises(ValueError, match='batch_size'):
        module.ResidualLiberoTrainer(module.TrainConfig('model', batch_size=0))


def test_training_releases_adapter_before_training(load_script, monkeypatch, tmp_path):
    import weakref

    module = load_script('scripts/train_residual_libero.py')
    references = []

    def make_adapter(config):
        adapter = Mock()
        references.append(weakref.ref(adapter))
        return adapter

    monkeypatch.setattr(module, '_make_adapter', make_adapter)
    monkeypatch.setattr(module.LiberoSampleCollector, 'collect', lambda self: ([], [{}]))
    trainer = module.ResidualLiberoTrainer(module.TrainConfig('model', output=tmp_path))
    bank = Mock()
    monkeypatch.setattr(module, 'RetrievalBank', Mock(return_value=bank))

    def train(*args):
        assert references[0]() is None
        return 'config', 'params'

    monkeypatch.setattr(trainer, 'train', train)
    monkeypatch.setattr(trainer, 'save_checkpoint', Mock())
    trainer.run()
    bank.save.assert_called_once_with(tmp_path / 'retrieval_bank')
    trainer.save_checkpoint.assert_called_once_with('config', 'params')


def test_argparse_defaults_and_overrides(load_script):
    train = load_script('scripts/train_residual_libero.py')
    args = train._parse_args(['--groot-model-path', 'model', '--batch-size', '3'])
    assert args.batch_size == 3
    assert args.replan_steps == 4
    assert args.horizon == 8


def test_server_forwards_checkpoint_metadata(load_script):
    module = load_script('scripts/serve_residual_libero.py')
    policy = module.ResidualLiberoRolloutPolicy.from_checkpoint.return_value
    policy.config = SimpleNamespace(action_dim=7, action_horizon=8, replan_steps=4)
    module.main(module.Args('model', Path('checkpoint'), host='localhost', port=1234))
    factory = module.websocket_policy_server.WebsocketPolicyServer
    assert factory.call_args.kwargs['metadata']['recommended_replan_steps'] == 4
    assert factory.call_args.kwargs['host'] == 'localhost'
    factory.return_value.serve_forever.assert_called_once()


@pytest.mark.parametrize('policy_type, expected_length', [('LAP', 2), ('RESIDUAL', 4)])
def test_simulator_action_phase_length(load_script, policy_type, expected_length):
    module = load_script('scripts/libero/main.py')
    evaluator = module.LiberoEvaluator(module.Args(policy_type=module.PolicyType(policy_type), replan_steps=2))
    actions = np.ones((4, 7))
    chunk = evaluator._action_chunk({'actions': actions, 'replan_steps': 2}, {})
    assert len(chunk) == expected_length
    np.testing.assert_array_equal(chunk[:, -1], -np.ones(expected_length))
    if policy_type == 'RESIDUAL':
        with pytest.raises(ValueError, match='server uses'):
            evaluator._action_chunk({'actions': actions, 'replan_steps': 3}, {})


def test_simulator_resets_each_episode_and_closes_environment(load_script, monkeypatch, tmp_path):
    module = load_script('scripts/libero/main.py')
    args = module.Args(policy_type=module.PolicyType.RESIDUAL, replan_steps=2,
                       num_steps_wait=0, num_trials_per_task=2,
                       video_out_path=str(tmp_path), results_out_path=str(tmp_path))
    env = Mock()
    env.set_init_state.return_value = {}
    env.step.side_effect = [({}, 0, done, {}) for done in [False, False, True] * 2]
    suite = Mock(n_tasks=1)
    suite.get_task_init_states.return_value = [0, 1]
    module.benchmark.get_benchmark_dict.return_value = {'libero_10': lambda: suite}
    monkeypatch.setattr(module, '_get_libero_env', lambda *args: (env, 'move'))
    monkeypatch.setattr(module, 'get_images_from_obs', lambda *args: (np.zeros((2, 2, 3)), np.zeros((2, 2, 3))))
    monkeypatch.setattr(module, 'obs_to_request', lambda *args, **kwargs: kwargs)
    client = module._websocket_client_policy.WebsocketClientPolicy.return_value
    client.infer.side_effect = lambda request: {'actions': np.zeros((2, 7)), 'replan_steps': 2}
    result = module.LiberoEvaluator(args).run()
    assert result['summary']['total_successes'] == 2
    assert [r['global_episode_id'] for r in result['episodes']] == [0, 1]
    assert [call.args[0]['reset_residual_plan'] for call in client.infer.call_args_list] == [True, False, True, False]
    env.close.assert_called_once()
    saved = json.loads(next(tmp_path.glob('results_*.json')).read_text())
    assert saved['summary'] == result['summary']


def test_simulator_closes_environment_on_episode_error(load_script, monkeypatch):
    module = load_script('scripts/libero/main.py')
    evaluator = module.LiberoEvaluator(module.Args(num_trials_per_task=1))
    evaluator.task_suite = Mock()
    evaluator.task_suite.get_task_init_states.return_value = [0]
    env = Mock()
    monkeypatch.setattr(module, '_get_libero_env', lambda *args: (env, 'move'))
    monkeypatch.setattr(evaluator, '_run_episode', Mock(side_effect=RuntimeError('failed')))
    with pytest.raises(RuntimeError, match='failed'):
        evaluator._run_task(0)
    env.close.assert_called_once()
