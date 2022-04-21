# LoLGym for COMS4995 Deep Learning Project

## About

This repo contains code to train an agent to play league of legends using the PPO algorithm. In this project, we built an AI capable of playing and winning in the
game League of Legends, a popular multiplayer game created by Riot Games. We implement a model that uses the existing deep reinforcement learning practice of Proximal Policy Optimization (PPO) alongside an adversarial multi-agent architecture to outperform existing hard-coded bots and human players in a 1v1 game.

## Installation

### Install .NET

```shell
wget -q https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
add-apt-repository universe
apt-get update
apt-get install apt-transport-https
apt-get update
apt-get install dotnet-sdk-3.1
```

### Install the runtime

```shell
sudo apt-get update; \
  sudo apt-get install -y apt-transport-https && \
  sudo apt-get update && \
  sudo apt-get install -y dotnet-sdk-3.1
```

### Clone GameServer 
Download the 4.20 version of League game client

```shell
git clone https://github.com/MiscellaneousStuff/LeagueSandbox-RL-Learning
cd LeagueSandbox-RL-Learning && git checkout master && git branch && git submodule init && git submodule update
```

### Build GameServer
```shell
cd LeagueSandbox-RL-Learning  && dotnet build .
```

### Run game server to generate configs
```shell
cd /content/LeagueSandbox-RL-Learning/GameServerConsole/bin/Debug/netcoreapp3.0/ && \
/content/LeagueSandbox-RL-Learning/GameServerConsole/bin/Debug/netcoreapp3.0/GameServerConsole --redis_port 6379
```

### Install PyLOL
```shell
git clone -v https://github.com/jjlee0802cu/pylol.git
pip3 install --upgrade pylol/
```

### Install Redis Client and Redis Server
```shell
pip3 install redis
sudo apt-get install redis-server
```

### Write config dirs (Game Server)
```shell
touch PATH/TO/config_dirs.txt
printf "[dirs]\ngameserver = PATH/TO/LeagueSandbox-RL-Learning/GameServerConsole/bin/Debug/netcoreapp3.0/\n" > PATH/TO/config_dirs.txt
```

### Download LoL client
```shell
!wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vkrdzSxTN6FPP7A9R65bIEBNefU7vDJL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1vkrdzSxTN6FPP7A9R65bIEBNefU7vDJL" -O league-of-legends-420.tar.gz && rm -rf /tmp/cookies.txtInstall Winetricks
```

### Install Wine
```shell
sudo dpkg --add-architecture i386
sudo apt update -y
sudo apt install wine64 wine32 -y
```

### Install Winetricks
Without this you will get into the game, but your screen will be black.
```shell
sudo apt-get install winetricks -y
winetricks d3dx9
```

### Extract LoL Client
```shell
tar -xzvf league-of-legends-420.tar.gz
```

### Run Client
```shell
pip install pyautogui
pip install Xlib
```

### Write config dirs (LoL Client)
```shell
touch PATH/TO/config_dirs.txt
printf "lolclient = /content/League-of-Legends-4-20/RADS/solutions/lol_game_client_sln/releases/0.0.1.68/deploy/" > config_dirs.txt
```

### Install LolGym
```shell
git clone https://github.com/jjlee0802cu/lolgym.git
pip3 install -e lolgym/
```

## Usage

### PPO Agent trained against hard-coded scripted intermediate bot (Milestone)
Use commit 0c4958342468c1673e1f86046114b629f0268328
```shell
python3 ./lolgym/examples/full_game_ppo.py --epochs 200 --host <public_ip> --config_path "PATH/TO/config_dirs.txt" --run_client
```

### Two PPO Agents trained against each other using adversarial multi-agent architecture (Final report)
Use commit 5031243750aaeca52dd4f2d310681a90554cdeda
```shell
python3 ./lolgym/examples/full_game_ppo.py --epochs 200 --host <public_ip> --config_path "PATH/TO/config_dirs.txt" --run_client
```
