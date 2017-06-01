all:
	g++ -O3 plm_train.cpp src/Frame/pc_frame.cpp src/Utils/utils.cpp -I . -std=c++0x -o bin/plm_train -lpthread
	g++ -O3 plm_predict.cpp src/Frame/pc_frame.cpp src/Utils/utils.cpp -I . -std=c++0x -o bin/plm_predict -lpthread
