#ifndef __3X_ADD_1_H
#define __3X_ADD_1_H

#include <stdint.h>
#include <vector>
#include <cassert>

//a*x+b型
struct Pair {
    Pair():a(0),b(0){}
    Pair(uint64_t A,uint64_t B):a(A),b(B){}
    uint64_t a;
    uint64_t b;
};

#define MAX_ORDER 5

//3^1,3^2,3^3,3^4,3^5,...
extern std::vector<Pair> g_array[MAX_ORDER];

inline uint64_t log3(uint64_t n) {
    uint64_t res = 0;
    while (n > 1) {
        n = n / 3;
        res += 1;
    }
    return res;
}

bool pushPair(const Pair& pr);

void oneGenerate(Pair pr);

void three_add_one();


//////////////////////生成一个平面图/////////////////////
enum class CmdType {
    ADD_EDGE,
    REMOVE_EDGE,
    ADD_NODE,
    REMOVE_NODE,
};
// 
struct Cmd {
    CmdType type;
    int nodeId;
    int nodeId2;
};

struct CNode {
    bool isValid;
    int degree;
    int color;

    //tmp
    bool traveled = false;
};

//添加一个点
//添加一些边
class PlanarGraph {
public:
    void addEdge(int i, int j) {
        bool ok = addEdgeInternal(i, j);
        if (ok) {
            Cmd cmd{ CmdType::ADD_EDGE,i,j };
            revokeStack.push_back(cmd);
        }
    }
    
    void removeEdge(int i, int j) {
        bool ok = removeEdgeInternal(i, j);
        if (ok) {
            Cmd cmd{ CmdType::REMOVE_EDGE,i,j };
            revokeStack.push_back(cmd);
        }
    }

    void addNode(int i) {
        bool ok = addNodeInternal(i);
        if (ok) {
            revokeStack.push_back({ CmdType::ADD_NODE,i,0 });
        }
    }

    void removeNode(int i) {
        bool ok = removeNodeInternal(i);
        if (ok == true) {
            revokeStack.push_back({ CmdType::REMOVE_NODE,i,0 });
        }
        int siz = table.size();
        for (int j = 0; j < siz; ++j) {
            bool ok1 = removeEdgeInternal(j, i);
            if (ok1) {
                revokeStack.push_back({ CmdType::REMOVE_EDGE,i,j });
            }
        }
    }

    //点着色不支持撤销恢复也无需支持
    void setColor(int i ,int color) {
        if (nodes[i].isValid) {
            nodes[i].color = color;
        }
    }

    std::vector<int> subGraph(int v, int color1, int color2) {
        for (int i = 0; i < nodes.size(); ++i) {
            nodes[i].traveled = false;
        }
        return subGraphInternal(v, color1, color2);
    }

    //撤销上一次操作
    void revoke() {
        if (!revokeStack.empty()) {
            Cmd cmd = revokeStack.back();
            recoverStack.push_back(cmd);
            revokeStack.pop_back();
            invAct(cmd);
        }
    }

    //恢复上一次撤销的操作
    void recover() {
        if (!recoverStack.empty()) {
            Cmd cmd = recoverStack.back();
            revokeStack.push_back(cmd);
            recoverStack.pop_back();
            act(cmd);
        }
    }

    void act(const Cmd& cmd) {
        if (cmd.type == CmdType::ADD_EDGE) {
            addEdgeInternal(cmd.nodeId, cmd.nodeId2);
        }
        else if (cmd.type == CmdType::REMOVE_EDGE) {
            removeEdgeInternal(cmd.nodeId, cmd.nodeId2);
        }
        else if (cmd.type == CmdType::ADD_NODE) {
            addNodeInternal(cmd.nodeId);
        }
        else if (cmd.type == CmdType::REMOVE_NODE) {
            removeNodeInternal(cmd.nodeId);
        }
        else {

        }
    }

    void invAct(const Cmd& cmd) {
        if (cmd.type == CmdType::ADD_EDGE) {
            removeEdgeInternal(cmd.nodeId, cmd.nodeId2);
        }
        else if (cmd.type == CmdType::REMOVE_EDGE) {
            addEdgeInternal(cmd.nodeId, cmd.nodeId2);
        }
        else if (cmd.type == CmdType::ADD_NODE) {
            removeNodeInternal(cmd.nodeId);
        }
        else if (cmd.type == CmdType::REMOVE_NODE) {
            addNodeInternal(cmd.nodeId);
        }
        else {

        }
    }

    //生成四色着色
    void fourColorShader() {
        int times = 0;
        std::vector<int> removedNodes;
        while (nodeCount > 5) {
            bool bingo = false;
            for (int i = 0; i < nodes.size(); ++i) {
                if (nodes[i].isValid && nodes[i].degree <= 5) {
                    removeNode(i);
                    removedNodes.push_back(i);
                    times += 1;
                    bingo = true;
                    break;
                }
            }
            assert(bingo == true);
        }
        //执行<=5的着色


        //逻辑判断
        //计算|C(v_i)|
        bool color[4] = { 0,0,0,0 };
        for (int k = removedNodes.size() - 1; k >= 0; --k) {
            int id = removedNodes[k];
            revoke();
            for (int j = 0; j < table.size(); ++j) {
                if (nodes[j].isValid && j != id && table[id][j] > 0) {
                    color[nodes[j].color] = true;
                }
            }
            int n = 0;
            for (int i = 0; i < 4; ++i) {
                if (color[i] == true)n += 1;
            }

            if (n <= 3) {
                for (int i = 0; i < 4; ++i) {
                    if (color[i] == false) {
                        setColor(id, i);
                        break;
                    }
                }
                continue;
            }
            else if (n == 4 && nodes[id].degree == 4) {

            }
            else if( n==4 && nodes[id].degree == 5){
            }
            else {
                assert(false);
            }
        }

    }
protected:
    bool addEdgeInternal(int i,int j) {
        if (i == j)return false;
        if (nodes[i].isValid == false || nodes[j].isValid == false) return false;
        if (table[i][j] == 0) {
            table[i][j] = 1;
            nodes[i].degree += 1;
            table[j][i] = 1;
            nodes[j].degree += 1;
            return true;
        }
        return false;
    }

    bool removeEdgeInternal(int i, int j) {
        if (i == j)return false;
        if (nodes[i].isValid == false || nodes[j].isValid == false) return false;
        if (table[i][j] > 0) {
            table[i][j] = 0;
            nodes[i].degree -= 1;
            table[j][i] = 0;
            nodes[j].degree -= 1;
            return true;
        }
        return false;
    }

    bool addNodeInternal(int i) {
        if (nodes[i].isValid == false) {
            nodes[i].isValid = true;
            nodeCount += 1;
            return true;
        }
        return false;
    }

    bool removeNodeInternal(int i) {
        if (nodes[i].isValid == true) {
            nodes[i].isValid = false;
            nodeCount -= 1;
            return true;
        }
        return false;
    }

    //计算肯普链，和顶点i连通的所有着色为a,b的顶点集
    std::vector<int> subGraphInternal(int v, int color1, int color2) {
        if (nodes[v].isValid == false || nodes[v].color != color1 && nodes[v].color != color2) {
            return {};
        }
        std::vector<int> res;
        for (int j = 0; j < table.size(); ++j) {
            if (nodes[j].isValid && j != v && (nodes[j].color == color1 || nodes[j].color == color2) && nodes[j].traveled == false) {
                res.push_back(j);
                nodes[j].traveled = true;
            }
        }
        for (int k = 0; k < res.size(); ++k) {
            std::vector<int> tmp = subGraph(res[k], color1, color2);
            res.insert(res.end(), tmp.begin(), tmp.end());
        }
    }

    int nodeCount = 0;
    std::vector<CNode> nodes;
    std::vector<std::vector<int>> table;

    //撤销恢复栈
    //撤销栈
    std::vector<Cmd> revokeStack;
    //恢复栈
    std::vector<Cmd> recoverStack;
};

#endif