cd ..\gym-fx
git pull
python -m build
pip install .
cd ..\rl-optimizer
git pull
rl-optimizer.bat tests\data\x_d2_original.csv