from setuptools import setup, find_packages

setup(
    name='rl-optimizer',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'rl_optimizer=app.main:main'
        ],
        'rl_optimizer.optimizers': [
            'openrl=app.plugins.optimizer_plugin_openrl:Plugin'
        ],
        'rl_optimizer.environments': [
            'prediction=app.plugins.environment_plugin_prediction:Plugin',
        ],
        'rl_optimizer.agents': [
            'openrl_ppo=app.plugins.agent_plugin_openrl_ppo:Plugin',
            'dummy_automation=app.plugins.agent_plugin_dummy_automation:Plugin'
        ]
    },
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'openrl',
        'gym'
    ],
    author='Harvey Bastidas',
    author_email='your.email@example.com',
    description='A reinforcement learning optimization system that supports dynamic loading of optimizer, environment, and agent plugins for processing and optimizing trading strategies.',
    url='https://github.com/harveybc/rl-optimizer',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
