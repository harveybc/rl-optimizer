from setuptools import setup, find_packages

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='rl-optimizer',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'rl_optimizer=app.main:main'
        ],
        'rl_optimizer.optimizers': [
            'openrl=app.plugins.optimizer_plugin_openrl:Plugin',
            'neat=app.plugins.optimizer_plugin_neat:Plugin',
            'neat_p2p=app.plugins.optimizer_plugin_neat_p2p:Plugin',
            'openrl_optimizer=app.plugins.optimizer_plugin_openrl:Plugin'  # Example additional optimizer
        ],
        'rl_optimizer.environments': [
            'prediction=app.plugins.environment_plugin_prediction:Plugin',
            # Add other environment plugins here
        ],
        'rl_optimizer.agents': [
            'openrl_ppo=app.plugins.agent_plugin_openrl_ppo:Plugin',
            'dummy_automation=app.plugins.agent_plugin_dummy_automation:Plugin',
            # Add other agent plugins here
        ]
    },
    install_requires=[
        'numpy>=1.18.0',
        'pandas>=1.0.0',
        'scikit-learn>=0.22.0',
        'openrl>=0.1.0',
        'gym>=0.17.0'
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'sphinx>=3.0.0',
            'black>=21.0.0',
            'isort>=5.0.0'
        ],
        'docs': [
            'sphinx>=3.0.0',
            'sphinx_rtd_theme>=0.5.0'
        ]
    },
    author='Harvey Bastidas',
    author_email='your.email@example.com',
    description='A reinforcement learning optimization system that supports dynamic loading of optimizer, environment, and agent plugins for processing and optimizing trading strategies.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/harveybc/rl-optimizer',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='reinforcement learning, optimization, trading, NEAT, OpenRL, plugin architecture',
    python_requires='>=3.6',
    include_package_data=True,  # Include non-code files specified in MANIFEST.in
    zip_safe=False,  # Set to False if your package needs to access data files
)
