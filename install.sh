python3 -m venv venv
./venv/bin/python3 -m pip install -r ./requirements.txt


echo "echo 'stating app...'" > ./run.sh
echo "./venv/bin/python3 main.py" >> ./run.sh

echo ""
echo ""
echo ""
echo "app installed!"
echo ""
echo "	* run 'bash run.sh' to start."
echo ""
echo "	* all program assets are containd with in this folder. just delete to uninstall."
