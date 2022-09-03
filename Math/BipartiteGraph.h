/*****************************************************************************
* @brief : 实现基本的图结构
* @author : acedtang
* @date : 2021/2/9
* @version : ver 1.0
* @inparam : 
* @outparam : 
*****************************************************************************/

#ifndef __GRAPH_H
#define __GRAPH_H

#include <set>
#include <unordered_map>
#include <unordered_set>
#include "assert.h"

namespace Sun
{
    template<class _Data>
    class GraphNode
    {
    public:
        _Data _data;
        //相连接的节点
        std::set<GraphNode<_Data>*> _nodeSet;
    };

#define GraphType_UnDirection 0
#define GraphType_Direction 1


    template<class _Data>
    class Graph
    {
    public:
        Graph(int type = 0) {
            _type = type;
        }
        void addNode(GraphNode<_Data>* n) {
            _nodes.insert(n);
        }
        void eraseNode(GraphNode<_Data>* n) {
            if (n == nullptr) return;
            auto t = _nodes.find(n);
            if (t == _nodes.end()) return;
            for (auto& it:n->_nodeSet) {
                if (it) {
                    it->_nodeSet.erase(n);
                }
            }
            _nodes.erase(n);
        }
        bool addEdge(GraphNode<_Data>* n1, GraphNode<_Data>* n2) {
            auto t1 = _nodes.find(n1);
            auto t2 = _nodes.find(n2);
            if (t1 == _nodes.end() || t2 == _nodes.end()) {
                return false;
            }
            if (_type & GraphType_Direction) {
                t1->_nodeSet->insert(t2);
            }
            else {
                t1->_nodeSet->insert(t2);
                t2->_nodeSet->insert(t1);
            }
        }
    public:
        std::set<GraphNode<_Data>*> _nodes;
        int _type;
    };

    //2分图的一种简单实现,直接用函数实现
    template<class _Face, class _Vertex>
    class BipartiteGraph
    {
    public:
        void addRelation(_Face* face, _Vertex* vertex) {
            if (face == nullptr || vertex == nullptr)
            {
                assert(false); return;
            }
            _graph_faces[face].insert(vertex);
            _graph_pts[vertex].insert(face);
        }
        void removeReleation(_Face* face, _Vertex* vertex) {
            if (face == nullptr || vertex == nullptr)
            {
                assert(false); return;
            }
            if (_graph_faces.find(face) == _graph_faces.end() ||
                _graph_pts.find(vertex) == _graph_pts.end()) {
                return;
            }
            _graph_faces[face].erase(vertex);
            _graph_pts[vertex].erase(face);
        }
        void clear() {
            _graph_faces.clear();
            _graph_pts.clear();
        }
        void addFace(_Face* face) {
            if (face == nullptr)return;
            _graph_faces[face];
        }
        void addVertex(_Vertex* vertex) {
            if (vertex == nullptr) return;
            _graph_pts[vertex];
        }
        void eraseFace(_Face* face) {
            if (_graph_faces.find(face) == _graph_faces.end()) {
                return;
            }
            for (auto& it : _graph_faces[face]) {
                _graph_pts[it].erase(face);
            }
            _graph_faces.erase(face);
        }
        void eraseVertex(_Vertex* vertex) {
            if (_graph_pts.find(vertex) == _graph_pts.end()) {
                return;
            }
            for (auto& it : _graph_pts[vertex]) {
                _graph_faces[it].erase(vertex);
            }
            _graph_pts.erase(vertex);
        }
        std::unordered_set<_Face*> outputFaces(_Vertex* vertex) {
            if (_graph_pts.find(vertex) == _graph_pts.end())
                return {};
            return _graph_pts[vertex];
        }
        std::unordered_set<_Vertex*> outputVertices(_Face* face) {
            if (_graph_faces.find(face) == _graph_faces.end())
                return {};
            return _graph_faces[face];
        }
        //if vertex not exist , wille create one ,pay attetion!if not, we 
        //can not return a ref
        std::unordered_set<_Face*>& getFacesSetRef(_Vertex* vertex) {
            return _graph_pts[vertex];
        }
        //the same
        std::unordered_set<_Vertex*>& getVerticesSetRef(_Face* face) {
            return _graph_faces[face];
        }
        bool isFaceExist(_Face* face) {
            return _graph_faces.find(face) != _graph_faces.end();
        }
        bool isVertexExist(_Vertex* vertex) {
            return _graph_pts.find(vertex) != _graph_pts.end();
        }
    protected:
        std::unordered_map<_Face*, std::unordered_set<_Vertex*>> _graph_faces;
        std::unordered_map<_Vertex*, std::unordered_set<_Face*>> _graph_pts;
    };

    ////////////////////////////////////////

}



#endif