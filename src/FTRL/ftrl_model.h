#ifndef FTRL_MODEL_H_
#define FTRL_MODEL_H_

#include <unordered_map>
#include <string>
#include <vector>
#include <mutex>
#include <iostream>
#include <cmath>
#include "../Utils/utils.h"

using namespace std;

//每一个特征维度的模型单元
class ftrl_model_unit
{
public:
    vector<double> u;
    vector<double> u_n;
    vector<double> u_z;
    vector<double> w;
    vector<double> w_n;
    vector<double> w_z;
    mutex mtx;
public:
    ftrl_model_unit(int piece_num, double u_mean, double u_stdev, double w_mean, double w_stdev)
    {
        u.resize(piece_num);
        u_n.resize(piece_num);
        u_z.resize(piece_num);
        for(int f = 0; f < piece_num; ++f)
        {
            u[f] = utils::gaussian(u_mean, u_stdev);
            u_n[f] = 0.0;
            u_z[f] = 0.0;
        }
        w.resize(piece_num);
        w_n.resize(piece_num);
        w_z.resize(piece_num);
        for(int f = 0; f < piece_num; ++f)
        {
            w[f] = utils::gaussian(w_mean, w_stdev);
            w_n[f] = 0.0;
            w_z[f] = 0.0;
        }
    }

    ftrl_model_unit(int piece_num, const vector<string>& modelLineSeg)
    {
        u.resize(piece_num);
        u_n.resize(piece_num);
        u_z.resize(piece_num);
        w.resize(piece_num);
        w_n.resize(piece_num);
        w_z.resize(piece_num);
        for(int f = 0; f < piece_num; ++f)
        {
            u[f] = stod(modelLineSeg[1 + f]);
            w[f] = stod(modelLineSeg[piece_num + 1 + f]);
            u_n[f] = stod(modelLineSeg[2 * piece_num + 1 + f]);
            w_n[f] = stod(modelLineSeg[3 * piece_num + 1 + f]);
            u_z[f] = stod(modelLineSeg[4 * piece_num + 1 + f]);
            w_z[f] = stod(modelLineSeg[5 * piece_num + 1 + f]);
        }
    }

    void reinit_u(double u_mean, double u_stdev)
    {
        int size = u.size();
        for(int f = 0; f < size; ++f)
        {
            u[f] = utils::gaussian(u_mean, u_stdev);
        }
    }

    void reinit_w(double w_mean, double w_stdev)
    {
        int size = w.size();
        for(int f = 0; f < size; ++f)
        {
            w[f] = utils::gaussian(w_mean, w_stdev);
        }
    }

    friend inline ostream& operator <<(ostream& os, const ftrl_model_unit& mu)
    {
        if(mu.u.size() > 0)
        {
            os << mu.u[0];
        }
        for(int f = 1; f < mu.u.size(); ++f)
        {
            os << " " << mu.u[f];
        }
        for(int f = 0; f < mu.w.size(); ++f)
        {
            os << " " << mu.w[f];
        }
        for(int f = 0; f < mu.u_n.size(); ++f)
        {
            os << " " << mu.u_n[f];
        }
        for(int f = 0; f < mu.w_n.size(); ++f)
        {
            os << " " << mu.w_n[f];
        }
        for(int f = 0; f < mu.u_z.size(); ++f)
        {
            os << " " << mu.u_z[f];
        }
        for(int f = 0; f < mu.w_z.size(); ++f)
        {
            os << " " << mu.w_z[f];
        }
        return os;
    }
};



class ftrl_model
{
public:
    ftrl_model_unit* muBias;
    unordered_map<string, ftrl_model_unit*> muMap;

    int piece_num;
    double u_stdev;
    double u_mean;
    double w_stdev;
    double w_mean;

public:
    ftrl_model(double _piece_num);
    ftrl_model(double _piece_num, double _u_mean, double _u_stdev, double _w_mean, double _w_stdev);
    ftrl_model_unit* getOrInitModelUnit(string index);
    ftrl_model_unit* getOrInitModelUnitBias();

    double get_uTx(const vector<pair<string, double> >& x, ftrl_model_unit& muBias, vector<ftrl_model_unit*>& theta, int f);
    double get_wTx(const vector<pair<string, double> >& x, ftrl_model_unit& muBias, vector<ftrl_model_unit*>& theta, int f);
    double get_uTx(const vector<pair<string, double> >& x, ftrl_model_unit& muBias, unordered_map<string, ftrl_model_unit*>& theta, int f);
    double get_wTx(const vector<pair<string, double> >& x, ftrl_model_unit& muBias, unordered_map<string, ftrl_model_unit*>& theta, int f);
    double getScore(const vector<pair<string, double> >& x, ftrl_model_unit& muBias, unordered_map<string, ftrl_model_unit*>& theta);
    void outputModel(ofstream& out);
    bool loadModel(ifstream& in);
    void debugPrintModel();

private:
    double get_uif(unordered_map<string, ftrl_model_unit*>& theta, const string& index, int f);
    double get_wif(unordered_map<string, ftrl_model_unit*>& theta, const string& index, int f);
private:
    mutex mtx;
    mutex mtx_bias;
};


ftrl_model::ftrl_model(double _piece_num)
{
    piece_num = _piece_num;
    u_mean = 0.0;
    u_stdev = 0.0;
    w_mean = 0.0;
    w_stdev = 0.0;
    muBias = NULL;
}

ftrl_model::ftrl_model(double _piece_num, double _u_mean, double _u_stdev, double _w_mean, double _w_stdev)
{
    piece_num = _piece_num;
    u_mean = _u_mean;
    u_stdev = _u_stdev;
    w_mean = _w_mean;
    w_stdev = _w_stdev;
    muBias = NULL;
}


ftrl_model_unit* ftrl_model::getOrInitModelUnit(string index)
{
    unordered_map<string, ftrl_model_unit*>::iterator iter = muMap.find(index);
    if(iter == muMap.end())
    {
        mtx.lock();
        ftrl_model_unit* pMU = new ftrl_model_unit(piece_num, u_mean, u_stdev, w_mean, w_stdev);
        muMap.insert(make_pair(index, pMU));
        mtx.unlock();
        return pMU;
    }
    else
    {
        return iter->second;
    }
}


ftrl_model_unit* ftrl_model::getOrInitModelUnitBias()
{
    if(NULL == muBias)
    {
        mtx_bias.lock();
        muBias = new ftrl_model_unit(piece_num, 0, 0, 0, 0);
        mtx_bias.unlock();
    }
    return muBias;
}


double ftrl_model::get_uTx(const vector<pair<string, double> >& x, ftrl_model_unit& muBias, vector<ftrl_model_unit*>& theta, int f)
{
    double result = 0;
    result += muBias.u[f];
    for(int i = 0; i < x.size(); ++i)
    {
        result += theta[i]->u[f] * x[i].second;
    }
    return result;
}


double ftrl_model::get_wTx(const vector<pair<string, double> >& x, ftrl_model_unit& muBias, vector<ftrl_model_unit*>& theta, int f)
{
    double result = 0;
    result += muBias.w[f];
    for(int i = 0; i < x.size(); ++i)
    {
        result += theta[i]->w[f] * x[i].second;
    }
    return result;
}


double ftrl_model::get_uTx(const vector<pair<string, double> >& x, ftrl_model_unit& muBias, unordered_map<string, ftrl_model_unit*>& theta, int f)
{
    double result = 0;
    result += muBias.u[f];
    for(int i = 0; i < x.size(); ++i)
    {
        result += get_uif(theta, x[i].first, f) * x[i].second;
    }
    return result;
}


double ftrl_model::get_wTx(const vector<pair<string, double> >& x, ftrl_model_unit& muBias, unordered_map<string, ftrl_model_unit*>& theta, int f)
{
    double result = 0;
    result += muBias.w[f];
    for(int i = 0; i < x.size(); ++i)
    {
        result += get_wif(theta, x[i].first, f) * x[i].second;
    }
    return result;
}


double ftrl_model::getScore(const vector<pair<string, double> >& x, ftrl_model_unit& muBias, unordered_map<string, ftrl_model_unit*>& theta)
{
    double result = 0;
    vector<double> uTx(piece_num);
    double max_uTx = numeric_limits<double>::lowest();
    for(int f = 0; f < piece_num; ++f)
    {
        uTx[f] = get_uTx(x, muBias, theta, f);
        if(uTx[f] > max_uTx) max_uTx = uTx[f];
    }
    double numerator = 0.0;
    double denominator = 0.0;
    for(int f = 0; f < piece_num; ++f)
    {
        uTx[f] -= max_uTx;
        uTx[f] = exp(uTx[f]);
        double wTx = get_wTx(x, muBias, theta, f);
        double s_wx = utils::sigmoid(wTx);
        numerator += uTx[f] * s_wx;
        denominator += uTx[f];
    }
    return numerator / denominator;
}


double ftrl_model::get_uif(unordered_map<string, ftrl_model_unit*>& theta, const string& index, int f)
{
    unordered_map<string, ftrl_model_unit*>::iterator iter = theta.find(index);
    if(iter == theta.end())
    {
        return 0.0;
    }
    else
    {
        return iter->second->u[f];
    }
}


double ftrl_model::get_wif(unordered_map<string, ftrl_model_unit*>& theta, const string& index, int f)
{
    unordered_map<string, ftrl_model_unit*>::iterator iter = theta.find(index);
    if(iter == theta.end())
    {
        return 0.0;
    }
    else
    {
        return iter->second->w[f];
    }
}


void ftrl_model::outputModel(ofstream& out)
{
    out << "bias " << *muBias << endl;
    for(unordered_map<string, ftrl_model_unit*>::iterator iter = muMap.begin(); iter != muMap.end(); ++iter)
    {
        out << iter->first << " " << *(iter->second) << endl;
    }
}


void ftrl_model::debugPrintModel()
{
    cout << "bias " << *muBias << endl;
    for(unordered_map<string, ftrl_model_unit*>::iterator iter = muMap.begin(); iter != muMap.end(); ++iter)
    {
        cout << iter->first << " " << *(iter->second) << endl;
    }
}


bool ftrl_model::loadModel(ifstream& in)
{
    string line;
    if(!getline(in, line))
    {
        return false;
    }
    vector<string> strVec;
    utils::splitString(line, ' ', &strVec);
    if(strVec.size() != 6 * piece_num + 1)
    {
        return false;
    }
    muBias = new ftrl_model_unit(piece_num, strVec);
    while(getline(in, line))
    {
        strVec.clear();
        utils::splitString(line, ' ', &strVec);
        if(strVec.size() != 6 * piece_num + 1)
        {
            return false;
        }
        string& index = strVec[0];
        ftrl_model_unit* pMU = new ftrl_model_unit(piece_num, strVec);
        muMap[index] = pMU;
    }
    return true;
}



#endif /*FTRL_MODEL_H_*/
