/*****************************************************************************
* @brief : Graph
* 抽象出数学中的图结构，可以具象化为状态机模型，
* 等
* @author : acedtang
* @date : 2021/3/28
* @version : ver 1.0
* @inparam :
* @outparam :
*****************************************************************************/
#ifndef __GRAPH_H
#define __GRAPH_H

#include <cassert>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <iterator>
#include "MatrixN.h"

namespace std
{
    template<class _T>
    struct hash<std::pair<_T,_T>>   
    {
        //这个const noexpect一定要写上去
        //template<class _T1,>
        size_t operator()(const std::pair<_T,_T>& rhs) const noexcept   
        {
            //当然，可以使用其他的方式来组合这个哈希值,
            //这里是cppreference里面的例子，产生的数够乱就行。
            return (std::hash<int>()(rhs.first)) ^ (std::hash<int>()(rhs.second) << 1); 
        }
    };
}

namespace Sun {

    //矩阵表示法
    class TableGraph
    {
    public:
        TableGraph() {

        }
        void addNode(int id) {
            int r = mat_.getNumRow();
            if (id >= r) {
                mat_.resize(id + 1);
            }
        }
        void addRelation(int i, int j) {
            if (i >= mat_.getNumRow() || j >= mat_.getNumRow())return;
            mat_.getData(i, j) = 1;
        }
    private:
        MatrixN<int> mat_;
    };

    template<class _T>
    struct PairHashFunc {
        std::size_t operator() (const std::pair<_T,_T>& pr) const 
        {
            return std::hash<int>()(pr.first) ^ std::hash<int>()(pr.second);
        }
    };

    template<class _NodeKey, class _NodeValue, class _EdgeKey ,class _EdgeValue,
        class _NodeHash = std::hash<_NodeKey> ,class _EdgeHash = std::hash<_EdgeKey>>
    class GraphBase {
    public:
        typedef _NodeKey NodeKey;
        typedef _NodeValue NodeValue;
        typedef _EdgeKey EdgeKey;
        typedef _EdgeValue EdgeValue;

        typedef GraphBase< _NodeKey, _NodeValue, _EdgeKey, _EdgeValue,    _NodeHash, _EdgeHash> My;
       
        typedef struct {
            _EdgeValue value;
            _NodeKey startNode;
            _NodeKey endNode;
        } EdgeStruct;
        typedef struct {
            _NodeValue value;
            std::set<_EdgeKey> outEdges;
            std::set<_EdgeKey> inEdges;
        } NodeStruct;
        typedef std::unordered_map<_NodeKey, NodeStruct, _NodeHash> NodesType;
        typedef std::unordered_map<_EdgeKey, EdgeStruct, _EdgeHash> EdgesType;

        struct ConstKeyNode :public std::pair<const _NodeKey, _NodeValue> {
            const _NodeKey& getKey() {
                return std::pair<const _NodeKey, _NodeValue>::first;
            }
            _NodeValue& getValue() {
                return std::pair<const _NodeKey, _NodeValue>::second;
            }
        };

        struct ConstKeyEdge :public std::pair<const _EdgeKey, _EdgeValue> {
            const _EdgeKey& getKey() {
                return std::pair<const _EdgeKey, _EdgeValue>::first;
            }
            _EdgeValue& getValue() {
                return std::pair<const _EdgeKey, _EdgeValue>::second;
            }
        };

        struct Node :public std::pair<_NodeKey, _NodeValue>
        {
            _NodeKey& getKey() {
                return std::pair<_NodeKey, _NodeValue>::first;
            }
            _NodeValue& getValue() {
                return std::pair<const _NodeKey, _NodeValue>::second;
            }
        };

        struct Edge :public std::pair<_EdgeKey, _EdgeValue>
        {
            _EdgeKey& getKey() {
                return std::pair<_EdgeKey, _EdgeValue>::first;
            }
            _EdgeValue& getValue() {
                return std::pair<_EdgeKey, _EdgeValue>::second;
            }
        };

        class iterator {
            friend class My;
        public:
            /*const Node& operator*() {
                return *iter_;
            }
            Node* operator->() {
                return iter_;
            }*/
            const _NodeKey& getKey() {
                return iter_->first;
            }
            _NodeValue& getValue() {
                return iter_->second.value;
            }

            const std::set<_EdgeKey>& getOutEdges(){
                return iter_->second.outEdges;
            }

            const std::set<_EdgeKey>& getInEdges(){
                return iter_->second.inEdges;
            }

            void setValue(const _NodeValue& value) {
                iter_->second.value = value;
            }
            bool operator== (const iterator& other) {
                return iter_ == other.iter_;
            }
            bool operator!=(const iterator& other) {
                return iter_ != other.iter_;
            }
            iterator& operator++() {
                ++iter_;
                return *this;
            }
            iterator& operator--() {
                --iter_;
                return *this;
            }
        private:
            typename NodesType::iterator iter_;
        };

    protected:
        std::unordered_map<_NodeKey, NodeStruct, _NodeHash> nodes_;
        std::unordered_map<_EdgeKey, EdgeStruct, _EdgeHash> edges_;
        std::map<std::pair<NodeKey, NodeKey>, _EdgeKey> nodePairs_;

        //functions
    public:
        void clear() {
            nodes_.clear();
            edges_.clear();
            nodePairs_.clear();
        }

        //if the key not exist ,will return a default value
        //O(1)
        _NodeValue getNodeValue(const _NodeKey& key) {
            auto it = nodes_.find(key);
            if (it != nodes_.end()) {
                std::pair<const _NodeKey, NodeStruct>& ptr = *it;
                
                return ptr.second.value;
            }
            return _NodeValue();
        }

        //this may return null if the key not exist
        _NodeValue* getNodeValuePtr(const _NodeKey& key) {
            auto it = nodes_.find(key);
            if (it != nodes_.end()) {
                std::pair<const _NodeKey, NodeStruct>& ptr = *it;

                return &ptr.second.value;
            }
            return nullptr;
        }

        //make sure the key exist,if not will crash 
        //O(1)
        const _EdgeValue& getEdgeValue(const _EdgeKey& key) {
            auto it = edges_.find(key);
            return it->second.value;
        }

        _EdgeValue* getEdgeValuePtr(const _EdgeKey& edge) {
            auto it = edges_.find(edge);
            if (it != edges_.end()) {
                return &(it->second.value);
            }
            return nullptr;
        }

        //make sure the key exist,if not will crash! 
        //O(1)
        const NodeKey& getEdgeStartNode(const _EdgeKey& key) {
            auto it = edges_.find(key);
            return (it->second.startNode);
        }

        //make sure the key exist,if not will crash!
        const NodeKey& getEdgeEndNode(const _EdgeKey& key) {
            auto it = edges_.find(key);
            return (it->second.endNode);
        }

        bool addNode(const _NodeKey& key,const _NodeValue& value = _NodeValue()) {
            auto it = nodes_.find(key);
            if (it != nodes_.end()) return false;

            NodeStruct stru;
            stru.value = value;
            nodes_.insert(std::make_pair(key, stru));

            return true;
        }

        //make sure the n1 and n2 are exist, if not ,will return false;
        bool addEdge(const _EdgeKey& key, const _EdgeValue& value, const _NodeKey& n1, const _NodeKey& n2) {
            auto iter1 = nodes_.find(n1);
            auto iter2 = nodes_.find(n2);
            if (iter1 == nodes_.end() || iter2 == nodes_.end()) {
                return false;
            }
            if (edges_.find(key) != edges_.end()) return false;

            EdgeStruct es;
            es.value = value;
            es.startNode = n1;
            es.endNode = n2;
            edges_.insert(std::make_pair(key, es));

            iter1->second.outEdges.insert(key);
            iter2->second.inEdges.insert(key);

            nodePairs_.insert(std::make_pair(std::make_pair(n1, n2), key));
            
            return true;
        }

        //the related edges will be removed
        bool removeNode(const _NodeKey& nodeKey) {
            auto iter = nodes_.find(nodeKey);
            if (iter == nodes_.end()) return false;

            NodeStruct stru = iter->second;//a cpy for iterator may invalid
            for (auto it : stru.outEdges) {
                removeEdge(it);
            }
            for (auto& it : stru.inEdges) {
                removeEdge(it);
            }
            nodes_.erase(nodeKey);

            return true;
        }

        bool removeEdge(const _EdgeKey& edgeKey) {
            auto it = edges_.find(edgeKey);
            if (it == edges_.end()) return false;

            EdgeStruct& stru = it->second;
            auto it2 = nodes_.find(stru.startNode);
            assert(it2 != nodes_.end());
            it2->second.outEdges.erase(edgeKey);

            auto it3 = nodes_.find(stru.endNode);
            assert(it3 != nodes_.end());
            it3->second.inEdges.erase(edgeKey);

            nodePairs_.erase(std::make_pair(it->second.startNode, it->second.endNode));
            edges_.erase(edgeKey);
            
            return true;
        }

        bool removeEdge(const _NodeKey& n1, const _NodeKey& n2) {
            auto it = nodes_.find(n1);
            if (it == nodes_.end() || nodes_.find(n2) == nodes_.end())
                return false;
            
            auto it2 = nodePairs_.find(std::make_pair(n1, n2));
            if (it2 == nodePairs_.end()) return false;

            _EdgeKey& key = it2->second;
            bool suc  = removeEdge(key);
            return suc;
        }

        size_t getNumChildren(const _NodeKey& node) {
            auto it = nodes_.find(node);
            if (it == nodes_.end()) {
                assert(false);
                return 0;
            }
            return it->second.outEdges.size();
        }

        std::vector<_NodeKey> getChildren(const _NodeKey& nodeKey) {
            std::vector<_NodeKey> res;

            auto it2 = nodes_.find(nodeKey);
            if (it2 == nodes_.end()) return {};
            
            NodeStruct& stru = it2->second;
            for (auto& it : stru.outEdges) {
                res.push_back(edges_[it].endNode);
            }

            return res;
        }

        std::vector<EdgeKey> getOutEdges(const _NodeKey& key) {
            std::vector<EdgeKey> edges;
            auto it = nodes_.find(key);
            if (it == nodes_.end()) return {};

            NodeStruct& stru = it->second;
            for (auto& iter : stru.outEdges) {
                edges.push_back(iter);
            }
            return edges;
        }

        size_t getNumNodes() {
            return nodes_.size();
        }

        size_t getNumEdges() {
            return edges_.size();
        }

        iterator begin() {
            iterator it;
            it.iter_ = nodes_.begin();
            return it;
        }

        iterator end() {
            iterator it;
            it.iter_ = nodes_.end();
            return it;
        }

        //decide whether this graph is a DFA
        //the outedge of the same node will have all different values
        bool isDFA() {
            for (iterator it = begin(); it != end(); ++it) {
                const std::set<EdgeKey>& ss= it.getOutEdges();
                std::set<EdgeValue> st;
                for (auto& it : ss) {
                    const EdgeValue& value = getEdgeValue(it);
                    auto pr = st.insert(value);
                    if (pr.second == false)
                        return false;
                }
                return true;
            }
        }
    
    };
}

#endif 

