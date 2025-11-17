"""
Integration tests to validate the F1 Racing RL system.

Run with: pytest tests/test_integration.py -v
Or: python tests/test_integration.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.envs.f1_racing_env import F1RacingEnv
from src.envs.domain_randomization import DomainRandomizer, create_randomized_env
from src.envs.opponent_ai import OpponentAI, OPPONENT_PROFILES
from src.physics.f1_car import F1Car, F1CarConfig
from src.physics.tire_model import TireModel, TireCompound
from src.tracks import get_circuit


def test_environment_creation():
    """Test basic environment creation."""
    print("\n" + "="*60)
    print("TEST: Environment Creation")
    print("="*60)

    env = F1RacingEnv(circuit_name='silverstone')

    assert env.action_space.shape == (6,), f"Action space shape incorrect: {env.action_space.shape}"
    assert len(env.observation_space.shape) == 1, "Observation should be 1D"

    print(f"✓ Action space: {env.action_space.shape}")
    print(f"✓ Observation space: {env.observation_space.shape}")
    print("✓ Environment created successfully")

    env.close()


def test_environment_reset():
    """Test environment reset."""
    print("\n" + "="*60)
    print("TEST: Environment Reset")
    print("="*60)

    env = F1RacingEnv(circuit_name='silverstone')
    obs, info = env.reset()

    assert obs.shape == env.observation_space.shape, "Observation shape mismatch"
    assert isinstance(info, dict), "Info should be a dictionary"

    print(f"✓ Observation shape: {obs.shape}")
    print(f"✓ Info keys: {list(info.keys())}")
    print("✓ Reset successful")

    env.close()


def test_environment_step():
    """Test environment step with random action."""
    print("\n" + "="*60)
    print("TEST: Environment Step")
    print("="*60)

    env = F1RacingEnv(circuit_name='silverstone')
    obs, _ = env.reset()

    # Random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == env.observation_space.shape, "Observation shape mismatch"
    assert isinstance(reward, (int, float, np.number)), f"Reward should be numeric, got {type(reward)}"
    assert isinstance(terminated, (bool, np.bool_)), "Terminated should be boolean"
    assert isinstance(truncated, (bool, np.bool_)), "Truncated should be boolean"

    print(f"✓ Action: {action}")
    print(f"✓ Observation shape: {obs.shape}")
    print(f"✓ Reward: {reward:.4f}")
    print(f"✓ Terminated: {terminated}, Truncated: {truncated}")
    print("✓ Step successful")

    env.close()


def test_episode_rollout():
    """Test a short episode rollout."""
    print("\n" + "="*60)
    print("TEST: Episode Rollout (100 steps)")
    print("="*60)

    env = F1RacingEnv(circuit_name='silverstone')
    obs, _ = env.reset()

    episode_reward = 0.0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            break

    print(f"✓ Completed {step + 1} steps")
    print(f"✓ Total reward: {episode_reward:.2f}")
    print(f"✓ Final distance: {info.get('distance', 0):.1f}m")
    print("✓ Rollout successful")

    env.close()


def test_physics_components():
    """Test individual physics components."""
    print("\n" + "="*60)
    print("TEST: Physics Components")
    print("="*60)

    # Test car
    car = F1Car()
    controls = {
        'throttle': 0.8,
        'brake': 0.0,
        'steering': 0.1,
        'gear': 3,
        'ers_mode': 0.5,
        'drs_active': True
    }
    state = car.step(controls, dt=0.02)

    print(f"✓ Car physics: Speed = {np.linalg.norm(car.velocity) * 3.6:.1f} km/h")

    # Test tires
    tire = TireModel(TireCompound.C3)
    tire.update(
        speed=50.0,
        lateral_accel=5.0,
        longitudinal_accel=3.0,
        steering_angle=0.1,
        brake_pressure=0.0,
        downforce=5000.0,
        track_temp=30.0,
        ambient_temp=20.0,
        dt=0.02
    )

    print(f"✓ Tire model: Avg temp = {np.mean(tire.temperatures):.1f}°C")
    print(f"✓ Tire grip: {tire.get_grip_coefficient():.2f}")
    print("✓ Physics components working")


def test_circuits():
    """Test circuit loading."""
    print("\n" + "="*60)
    print("TEST: Circuit Loading")
    print("="*60)

    circuits = ['silverstone', 'monaco', 'spa']

    for circuit_name in circuits:
        circuit = get_circuit(circuit_name)
        print(f"✓ {circuit.name}: {circuit.length:.0f}m, {circuit.num_corners} corners")

    print("✓ All circuits loaded successfully")


def test_domain_randomization():
    """Test domain randomization."""
    print("\n" + "="*60)
    print("TEST: Domain Randomization")
    print("="*60)

    randomizer = DomainRandomizer()
    params = randomizer.randomize()

    print(f"✓ Grip multiplier: {params['grip_multiplier']:.3f}")
    print(f"✓ Drag multiplier: {params['drag_multiplier']:.3f}")
    print(f"✓ Power multiplier: {params['power_multiplier']:.3f}")

    # Test wrapped environment
    base_env = F1RacingEnv(circuit_name='silverstone')
    wrapped_env = create_randomized_env(base_env, enable_randomization=True)

    obs, _ = wrapped_env.reset()
    action = wrapped_env.action_space.sample()
    obs, reward, terminated, truncated, info = wrapped_env.step(action)

    print("✓ Domain randomization working")
    wrapped_env.close()


def test_opponent_ai():
    """Test opponent AI."""
    print("\n" + "="*60)
    print("TEST: Opponent AI")
    print("="*60)

    circuit = get_circuit('silverstone')

    for profile_name, profile_config in OPPONENT_PROFILES.items():
        opponent = OpponentAI(profile_config)

        state = {
            'position': np.array([0.0, 0.0]),
            'velocity': np.array([50.0, 0.0]),
            'heading': 0.0
        }

        action = opponent.get_action(state, circuit)

        print(f"✓ {profile_name}: skill={profile_config.skill_level:.2f}, action shape={action.shape}")

    print("✓ Opponent AI working")


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("F1 RACING RL - INTEGRATION TESTS")
    print("="*70)

    tests = [
        test_environment_creation,
        test_environment_reset,
        test_environment_step,
        test_episode_rollout,
        test_physics_components,
        test_circuits,
        test_domain_randomization,
        test_opponent_ai,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ TEST FAILED: {test_func.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
