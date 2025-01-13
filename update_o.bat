
cd ..\neat-test
git pull
python -m build
pip install .
cd ..\rl-optimizer
git pull
rl-optimizer.bat tests\data\base_d2.csv