#include "3x_add_1.h"
#include <math.h>

std::vector<Pair> g_array[MAX_ORDER];

bool pushPair(const Pair& tmp) {
    uint64_t index = log3(tmp.a);
    if (index >= MAX_ORDER) {
        return false;
    }
    int siz = g_array[index].size();
    bool bingo = false;
    for (int i = 0; i < siz; ++i) {
        if (g_array[index][i].b == tmp.b) {
            bingo = true;
            break;
        }
    }
    if (bingo == false) {
        g_array[index].push_back(Pair(tmp.a, tmp.b));
    }
    return !bingo;
}

void oneGenerate(Pair pr) {
    Pair tmp;
    //a,b¶¼ÊÇÆæÊý
    if ((pr.a & 1)== 1 && (pr.b&1)==1) {
        //a*x+b -> 2a*x'+b -> 3a*x'+(3b+1)/2
        tmp.a = pr.a*3;
        tmp.b = (pr.b*3+1)/2;
        pushPair(tmp);

        //a*x+b -> a*(2x'+1)+b -> 2a * x'+(a+b) -> a*x'+(a+b)/2
        tmp.a = pr.a;
        tmp.b = (pr.a + pr.b)/2;
        pushPair(tmp);
    }
    else if ((pr.a & 1) == 1 && (pr.b & 1) == 0) {
        //a*x+b -> 2a*x'+b -> a*x'+b/2
        tmp.a = pr.a;
        tmp.b = pr.b / 2;
        pushPair(tmp);

        //a*x+b ->a*(2x'+1)+b -> 2a*x' +(a+b) -> 3a*x'+[3(a+b)+1]/2
        tmp.a = 3 * pr.a;
        tmp.b = (3 * (pr.a + pr.b)+1) / 2;
        pushPair(tmp);
    }

}

void three_add_one() {
    //the first is 3x+2
    Pair pr;
    pr.a = 3;
    pr.b = 2;
    g_array[1].push_back(pr);

    int ts = 1;

    int totalCount = 1;

    for (int i = 1; i < MAX_ORDER; ++i) {

    }
    while (true) {
        for (int i = 1; i < MAX_ORDER; ++i) {
            auto& vec = g_array[i];
            size_t siz = vec.size();
            for (int j = 0; j < siz; ++j) {
                oneGenerate(vec[j]);
            }
        }
        int count = 0;
        for (int i = 0; i < MAX_ORDER; ++i) {
            count += g_array[i].size();
        }
        if (count > totalCount) {
            totalCount = count;
        }
        else {
            break;
        }
    }
    

}