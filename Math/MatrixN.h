/*****************************************************************************
* @brief : MatrixN
* ʵ��һ��n*n�ľ���n����������,����Ԫ,��ʼΪ0*0����
* @author : acedtang
* @date : 2021/4/7
* @version : ver 1.0
* @inparam :
* @outparam :
*****************************************************************************/

#ifndef __MATRIXN_H
#define __MATRIXN_H

#include <vector>

namespace Sun {
    template<class _T>
    class MatrixN 
    {
    public:
        MatrixN() {
            numRow = 0;
        }
        MatrixN(int n) {
            datas_.resize(n * n);
            numRow = n;
        }
        void addRow() {
            numRow += 1;
            datas_.resize(numRow * numRow);
        }
        //��ȡ��i��,j��Ԫ��
        _T& getData(int i, int j) {
            return datas_[i * numRow + j];
        }
        void setData(int i, int j ,const _T& data) {
            datas_[i * numRow + j] = data;
        }
        int getNumRow() {
            return numRow;
        }
        //resize ֮��Ҫ����֮ǰԪ���ھ����е����λ�ò��䣬������λ�ñ任
        void resize(int n) {
            if (n == numRow) return;
            else if (n < numRow) {
                for (int r = 1; r < n; ++r) {
                    for (int j = 0; j < n; ++j) {
                        datas_[r * n + j] = datas_[r * numRow + j];
                    }
                }
                datas_.resize(n * n);
            }
            else {
                datas_.resize(n * n);
                for (int r = numRow-1; r>=1; --r) {
                    for (int j = 0; j < numRow; ++j) {
                        datas_[r * n + j] = datas_[r * numRow + j];
                    }
                }
            }
            numRow = n;
            return;
        }
    private:
        int numRow;
        std::vector<_T> datas_;
    };
}


#endif

