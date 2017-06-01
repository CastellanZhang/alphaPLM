#ifndef FTRL_TRAINER_H_
#define FTRL_TRAINER_H_

#include "../Frame/pc_frame.h"
#include "ftrl_model.h"
#include "../Sample/plm_sample.h"
#include "../Utils/utils.h"


struct trainer_option
{
    trainer_option() : u_bias(true), w_bias(true), piece_num(4), u_mean(0.0), u_stdev(0.1), w_mean(0.0), w_stdev(0.1), u_alpha(0.05), u_beta(1.0), u_l1(0.1), u_l2(5.0),
               w_alpha(0.05), w_beta(1.0), w_l1(0.1), w_l2(5.0),
               threads_num(1), b_init(false) {}
    string model_path, init_m_path;
    double u_mean, u_stdev, w_mean, w_stdev;
    double u_alpha, u_beta, u_l1, u_l2;
    double w_alpha, w_beta, w_l1, w_l2;
    int threads_num, piece_num;
    bool u_bias, w_bias, b_init;
    
    void parse_option(const vector<string>& args)
    {
        int argc = args.size();
        if(0 == argc) throw invalid_argument("invalid command\n");
        for(int i = 0; i < argc; ++i)
        {
            if(args[i].compare("-m") == 0) 
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                model_path = args[++i];
            }
            else if(args[i].compare("-u_bias") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                u_bias = (0 == stoi(args[++i])) ? false : true;
            }
            else if(args[i].compare("-w_bias") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_bias = (0 == stoi(args[++i])) ? false : true;
            }
            else if(args[i].compare("-piece_num") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                piece_num = stoi(args[++i]);
            }
            else if(args[i].compare("-u_stdev") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                u_stdev = stod(args[++i]);
            }
            else if(args[i].compare("-w_stdev") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_stdev = stod(args[++i]);
            }
            else if(args[i].compare("-w_alpha") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_alpha = stod(args[++i]);
            }
            else if(args[i].compare("-w_beta") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_beta = stod(args[++i]);
            }
            else if(args[i].compare("-w_l1") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_l1 = stod(args[++i]);
            }
            else if(args[i].compare("-w_l2") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                w_l2 = stod(args[++i]);
            }
            else if(args[i].compare("-u_alpha") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                u_alpha = stod(args[++i]);
            }
            else if(args[i].compare("-u_beta") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                u_beta = stod(args[++i]);
            }
            else if(args[i].compare("-u_l1") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                u_l1 = stod(args[++i]);
            }
            else if(args[i].compare("-u_l2") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                u_l2 = stod(args[++i]);
            }
            else if(args[i].compare("-core") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                threads_num = stoi(args[++i]);
            }
            else if(args[i].compare("-im") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command\n");
                init_m_path = args[++i];
                b_init = true; //if im field exits , that means b_init = true !
            }
            else
            {
                throw invalid_argument("invalid command\n");
                break;
            }
        }
    }

};


class ftrl_trainer : public pc_task
{
public:
    ftrl_trainer(const trainer_option& opt);
    virtual void run_task(vector<string>& dataBuffer);
    bool loadModel(ifstream& in);
    void outputModel(ofstream& out);
private:
    void train(int y, const vector<pair<string, double> >& x);
private:
    ftrl_model* pModel;
    double u_alpha, u_beta, u_l1, u_l2;
    double w_alpha, w_beta, w_l1, w_l2;
    bool u_bias;
    bool w_bias;
};


ftrl_trainer::ftrl_trainer(const trainer_option& opt)
{
    u_alpha = opt.u_alpha;
    u_beta = opt.u_beta;
    u_l1 = opt.u_l1;
    u_l2 = opt.u_l2;
    w_alpha = opt.w_alpha;
    w_beta = opt.w_beta;
    w_l1 = opt.w_l1;
    w_l2 = opt.w_l2;
    u_bias = opt.u_bias;
    w_bias = opt.w_bias;
    pModel = new ftrl_model(opt.piece_num, opt.u_mean, opt.u_stdev, opt.w_mean, opt.w_stdev);
}

void ftrl_trainer::run_task(vector<string>& dataBuffer)
{
    for(int i = 0; i < dataBuffer.size(); ++i)
    {
        plm_sample sample(dataBuffer[i]);
        train(sample.y, sample.x);
    }
}


bool ftrl_trainer::loadModel(ifstream& in)
{
    return pModel->loadModel(in);
}


void ftrl_trainer::outputModel(ofstream& out)
{
    return pModel->outputModel(out);
}


//输入一个样本，更新参数
void ftrl_trainer::train(int y, const vector<pair<string, double> >& x)
{
    ftrl_model_unit* thetaBias = pModel->getOrInitModelUnitBias();
    vector<ftrl_model_unit*> theta(x.size(), NULL);
    int xLen = x.size();
    for(int i = 0; i < xLen; ++i)
    {
        const string& index = x[i].first;
        theta[i] = pModel->getOrInitModelUnit(index);
    }
    vector<double> uTx(pModel->piece_num);
    vector<double> wTx(pModel->piece_num);
    double max_uTx = numeric_limits<double>::lowest();
    for(int f = 0; f < pModel->piece_num; ++f)
    {
        uTx[f] = pModel->get_uTx(x, *thetaBias, theta, f);
        wTx[f] = pModel->get_wTx(x, *thetaBias, theta, f);
        if(uTx[f] > max_uTx) max_uTx = uTx[f];
    }
    double denominator1 = 0.0;
    double denominator2 = 0.0;
    for(int f = 0; f < pModel->piece_num; ++f)
    {
        uTx[f] -= max_uTx;
        uTx[f] = exp(uTx[f]);
        wTx[f] = utils::sigmoid(y * wTx[f]);
        denominator1 += uTx[f];
        denominator2 += uTx[f] * wTx[f];
    }
    //update u_n, u_z
    for(int i = 0; i <= xLen; ++i)
    {
        ftrl_model_unit& mu = i < xLen ? *(theta[i]) : *thetaBias;
        double xi = i < xLen ? x[i].second : 1.0;
        if(i < xLen || u_bias)
        {
            for(int f = 0; f < pModel->piece_num; ++f)
            {
                mu.mtx.lock();
                double& uif = mu.u[f];
                double& u_nif = mu.u_n[f];
                double& u_zif = mu.u_z[f];
                double u_gif = xi * uTx[f] * (1.0/denominator1 - wTx[f]/denominator2);
                double u_sif = 1 / u_alpha * (sqrt(u_nif + u_gif * u_gif) - sqrt(u_nif));
                u_zif += u_gif - u_sif * uif;
                u_nif += u_gif * u_gif;
                mu.mtx.unlock();
            }
        }
    }
    //update w_n, w_z
    for(int i = 0; i <= xLen; ++i)
    {
        ftrl_model_unit& mu = i < xLen ? *(theta[i]) : *thetaBias;
        double xi = i < xLen ? x[i].second : 1.0;
        if(i < xLen || w_bias)
        {
            for(int f = 0; f < pModel->piece_num; ++f)
            {
                mu.mtx.lock();
                double& wif = mu.w[f];
                double& w_nif = mu.w_n[f];
                double& w_zif = mu.w_z[f];
                double w_gif = -y * xi * uTx[f] * wTx[f] * (1.0-wTx[f]) / denominator2;
                double w_sif = 1 / w_alpha * (sqrt(w_nif + w_gif * w_gif) - sqrt(w_nif));
                w_zif += w_gif - w_sif * wif;
                w_nif += w_gif * w_gif;
                mu.mtx.unlock();
            }
        }
    }
    //update u via FTRL
    for(int i = 0; i <= xLen; ++i)
    {
        ftrl_model_unit& mu = i < xLen ? *(theta[i]) : *thetaBias;
        if(i < xLen || u_bias)
        {
            for(int f = 0; f < pModel->piece_num; ++f)
            {
                mu.mtx.lock();
                double& uf = mu.u[f];
                double& u_nf = mu.u_n[f];
                double& u_zf = mu.u_z[f];
                if(fabs(u_zf) <= u_l1)
                {
                    uf = 0.0;
                }
                else
                {
                    uf = (-1) *
                        (1 / (u_l2 + (u_beta + sqrt(u_nf)) / u_alpha)) *
                        (u_zf - utils::sgn(u_zf) * u_l1);
                }
                mu.mtx.unlock();
            }
        }
    }
    //update w via FTRL
    for(int i = 0; i <= xLen; ++i)
    {
        ftrl_model_unit& mu = i < xLen ? *(theta[i]) : *thetaBias;
        if(i < xLen || w_bias)
        {
            for(int f = 0; f < pModel->piece_num; ++f)
            {
                mu.mtx.lock();
                double& wf = mu.w[f];
                double& w_nf = mu.w_n[f];
                double& w_zf = mu.w_z[f];
                if(fabs(w_zf) <= w_l1)
                {
                    wf = 0.0;
                }
                else
                {
                    wf = (-1) *
                        (1 / (w_l2 + (w_beta + sqrt(w_nf)) / w_alpha)) *
                        (w_zf - utils::sgn(w_zf) * w_l1);
                }
                mu.mtx.unlock();
            }
        }
    }
    //////////
    //pModel->debugPrintModel();
    //////////
}


#endif /*FTRL_TRAINER_H_*/
