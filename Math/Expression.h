/*****************************************************************************
* @brief : Expression
* 表达式转换器，将数学表达式转化为DAG表示的内部结构
* 目前支持,+,-,*,/,sqrt(),sin,cos,exp,log,min,max,abs
* @author : acedtang
* @date : 2021/6/1
* @version : ver 1.0
* @inparam :
* @outparam :
*****************************************************************************/

#ifndef __EXPRESSION_H
#define __EXPRESSION_H

#include <vector>
#include <string>
#include <unordered_map>
#include <functional>

namespace Sun {

	enum OpCode {
        OP_ADD = 0,
        OP_SUB = 1,
		OP_MUL = 2,
		OP_DIV = 3,
		OP_POW = 4,
		OP_MIN = 5,
		OP_MAX = 6,
		OP_CLAD = 7,
		OP_CLAU = 8,
		OP_ATAN = 9,

		OP_SQRT = 20,
		OP_ABS = 21,
		OP_SIN = 22,
		OP_COS = 23,
		OP_EXP = 24,
		OP_LN = 25,
		
		OP_LEFT = 40,
		OP_RIGHT = 41,

		//for use by lexer
		OP_INT = 60,
		OP_FLOAT = 61,	
		OP_SYM = 62,
		OP_X = 63,  
		OP_Y = 64,
		OP_Z = 65,
	};

	//优先级映射表
	extern std::unordered_map<OpCode, int> g_priorityTable;
	//字符映射表
	extern std::unordered_map<OpCode, const char*> g_signTable;

	//a<b -1 a==b 0 a>b 1
	inline char comparePriority(const OpCode& a, const OpCode& b) {
		int pa = g_priorityTable[a];
		int pb = g_priorityTable[b];
		if (pa < pb)
			return -1;
		else if (pa > pb)
			return 1;
		else
			return 0;
		return 0;
	}

	//运算符节点
	struct ExpNode {
		unsigned char opCode;
		unsigned char left;
		unsigned char right;
		unsigned char output;
		float constant;
	};

	//why can not add a constructor？
	struct Token {
		OpCode code;
		union {
			int integer;
			float real;
			char* sym; //point to a key 
			char var;	
		} data;
		int left = -1;
		int right = -1;
		float value = 0.;

		int inputParamNum() {
			if (code < OP_SQRT) {
				return 2;
			}
			return 1;
		}
	};

	inline bool isNumber(const Token& tk) {
		return (tk.code == OP_FLOAT || tk.code == OP_SYM || tk.code == OP_X
			|| tk.code == OP_Y || tk.code == OP_Z );
	}

	inline bool isUnary(const Token& tk) {
		return (tk.code == OP_SQRT || tk.code == OP_ABS || 
		 tk.code == OP_SIN || tk.code == OP_COS || tk.code == OP_EXP || tk.code == OP_LN);
	}

	inline bool isBinary(const Token& tk) {
		return tk.code == OP_ADD || tk.code == OP_SUB || tk.code == OP_MUL || tk.code == OP_DIV || tk.code == OP_POW
			|| tk.code == OP_CLAD || tk.code == OP_CLAU || tk.code == OP_MAX || tk.code == OP_MIN || tk.code == OP_ATAN;
	}

	inline bool isOperator(const Token& tk) {
		return isUnary(tk) || isBinary(tk);
	}

	inline bool tokenEqual(const Token& t1, const Token& t2) {
		if (t1.code != t2.code) return false;
		if (isUnary(t1)) {
			return (t1.left == t2.left);
		}
		else if (isBinary(t1)) {
			return (t1.left == t2.left) && (t1.right == t2.right);
		}
		else if (t1.code==OP_X||t1.code==OP_Y||t1.code==OP_Z) {
			return true;
		}
		else if (t1.code == OP_SYM) {
			return t1.data.sym == t2.data.sym;
		}
		else if (t1.code == OP_FLOAT) {
			return t1.data.real == t2.data.real;
		}
		return false;
	}

	inline bool isSpace(char sym) {
		return sym == ' ';
	}

	inline bool isNum(char sym) {
		return '0' <= sym && sym <= '9';
	}

	inline bool isLetter(char sym) {
		return ('a' <= sym && sym <= 'z' || 'A' <= sym && sym <= 'Z');
	}
	inline bool isUpperCaseLetter(char sym) {
		return ('A' <= sym) && (sym <= 'Z');
	}
	inline bool isLowerCaseLetter(char sym) {
		return ('a' <= sym) && (sym <= 'z');
	}

	struct SymHash {
		std::size_t operator() (char* key) const
		{
		    return std::hash<std::string>()(key);
		}
	};

	struct Equal {
		bool operator()(char* left, char* right) const {
			return (strcmp(left, right) == 0);
		}
	};
	
	class SymsContainer {
	public:
		SymsContainer() {}
		//SymsContainer(const SymsContainer& other);
		SymsContainer(SymsContainer&& other);
		SymsContainer& operator=(SymsContainer&& other);
		~SymsContainer();
		float getValue(const char* key);
		char* setValue(const char* data,float value);
		void clear();
	protected:
		std::unordered_map<char*,float, SymHash, Equal> syms_;
	};

	class AExpression {
	public:
		AExpression();
		AExpression(AExpression&& r);
		AExpression(const char* expression);

		AExpression compose(const AExpression& other);
		void setExpression(const char* expression);
		float value(float x, float y, float z);
		void setSymValue(const char* sym, float value);
		
		const std::vector<Token>& getDAG() const;
		std::vector<Token> getSymValuedDAG();

		void clear();

		std::string oriStr() const;

		//friend Expression operator+(const Expression& e1, const Expression& e2);
	protected:
		//提取单词元组
		bool lexer(const char* expression);
		//中缀token表达式->前缀token表达式
		bool toPrefix();
		//前缀token表达式->DAG
		bool toDAG();

		void printPrefixTokens();
		void printDAG();
		void printToken(const Token& tk);
		
		bool init(const char* expression);
		//原始字符串
		std::string str_;
		//中缀token表达式
		std::vector<Token> tokens_;
		//前缀token表达式
		std::vector<Token> prefixTokens_;
		//表达式的DAG表示
		std::vector<Token> dag_;
		//符号常量存储表
		SymsContainer symsTable_;
	};

}

#endif