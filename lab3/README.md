# MAC
brew install swig
# LINUX
sudo apt-get install swig3.0

pip install pygame
pip install gymnasium
pip install 'gymnasium[box2d]'

##### Esto de abajo no me anduvo

  !pip3 install cmake gymnasium scipy numpy gymnasium[box2d] pygame==2.6.0 swig

Tal vez tengan que ejecutar lo siguiente en sus máquinas (ubuntu 20.04)
  sudo apt-get remove swig
  sudo apt-get install swig3.0
  sudo ln -s /usr/bin/swig3.0 /usr/bin/swig

En windows tambien puede ser necesario MSVC++

