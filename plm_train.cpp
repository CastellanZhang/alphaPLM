#include <iostream>
#include <map>
#include <fstream>
#include "src/Frame/pc_frame.h"
#include "src/FTRL/ftrl_trainer.h"

using namespace std;

string train_help() 
{
    return string(
            "\nusage: cat sample | ./plm_train [<options>]"
            "\n"
            "\n"
            "options:\n"
            "-m <model_path>: set the output model path\n"
            "-u_bias <u_bias>: if u_bias = 1, add bias term for u; if = 0, no bias term added\tdefault:1\n"
            "-w_bias <w_bias>: if w_bias = 1, add bias term for w; if = 0, no bias term added\tdefault:1\n"
            "-piece_num <piece_num>: division number\tdefault:4\n"
            "-u_stdev <u_stdev>: stdev for initialization of u\tdefault:0.1\n"
            "-w_stdev <w_stdev>: stdev for initialization of w\tdefault:0.1\n"
            "-u_alpha <u_alpha>: u is updated via FTRL, alpha is one of the learning rate parameters\tdefault:0.05\n"
            "-u_beta <u_beta>: u is updated via FTRL, beta is one of the learning rate parameters\tdefault:1.0\n"
            "-u_l1 <u_L1_reg>: L1 regularization parameter of u\tdefault:0.1\n"
            "-u_l2 <u_L2_reg>: L2 regularization parameter of u\tdefault:5.0\n"
            "-w_alpha <w_alpha>: w is updated via FTRL, alpha is one of the learning rate parameters\tdefault:0.05\n"
            "-w_beta <w_beta>: w is updated via FTRL, beta is one of the learning rate parameters\tdefault:1.0\n"
            "-w_l1 <w_L1_reg>: L1 regularization parameter of w\tdefault:0.1\n"
            "-w_l2 <w_L2_reg>: L2 regularization parameter of w\tdefault:5.0\n"
            "-core <threads_num>: set the number of threads\tdefault:1\n"
            "-im <initial_model_path>: set the initial value of model\n"
    );
}

vector<string> argv_to_args(int argc, char* argv[]) 
{
    vector<string> args;
    for(int i = 1; i < argc; ++i)
    {
        args.push_back(string(argv[i]));
    }
    return args;
}


int main(int argc, char* argv[])
{
    cin.sync_with_stdio(false);
    cout.sync_with_stdio(false);
    srand(time(NULL));
    trainer_option opt;
    try
    {
        opt.parse_option(argv_to_args(argc, argv));
    }
    catch(const invalid_argument& e)
    {
        cout << "invalid_argument:" << e.what() << endl;
        cout << train_help() << endl;
        return EXIT_FAILURE;
    }

    ftrl_trainer trainer(opt);

    if(opt.b_init) 
    {
        ifstream f_temp(opt.init_m_path.c_str());
        if(!trainer.loadModel(f_temp)) 
        {
            cout << "wrong model" << endl;
            return EXIT_FAILURE;
        }
        f_temp.close();
    }

    pc_frame frame;
    frame.init(trainer, opt.threads_num);
    frame.run();

    ofstream f_model(opt.model_path.c_str(), ofstream::out);
    trainer.outputModel(f_model);
    f_model.close();

    return 0;
}

