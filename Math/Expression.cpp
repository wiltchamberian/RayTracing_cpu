#include "Expression.h"
#include "Graph.h"
#include <iostream>
#include <set>
#include <cmath>

namespace Sun {

	std::unordered_map<OpCode, int> g_priorityTable =
	{
		{OP_ADD,0},
		{OP_SUB,0},
		{OP_MUL,1},
		{OP_DIV,1},
		{OP_POW,2},
		{OP_MIN,3},
		{OP_MAX,3},
		{OP_CLAD,3},
		{OP_CLAU,3},
		{OP_ATAN,3},

		{OP_SQRT,10},
		{OP_ABS,10},
		{OP_SIN,10},
		{OP_COS,10},
		{OP_EXP,10},
		{OP_LN,10},
		{OP_CLAD,10},
		{OP_CLAU,10},
		
		{OP_LEFT,20},
		{OP_RIGHT,20}
	};

	std::unordered_map<OpCode, const char*> g_signTable = {
		{OP_ADD,"+"},
		{OP_SUB,"-"},
		{OP_MUL,"*"},
		{OP_DIV,"/"},
		{OP_POW,"^"},
		{OP_CLAD,"clad"},
		{OP_CLAU,"clau"},
		{OP_ATAN,"atan"},

		{OP_SQRT,"sqrt"},
		{OP_ABS,"abs"},
		{OP_MIN,"min"},
		{OP_MAX,"max"},
		{OP_SIN,"sin"},
		{OP_COS,"cos"},
		{OP_EXP,"exp"},
		{OP_LN,"ln"},
		{OP_ATAN,"atan"},

		{OP_LEFT,"("},
		{OP_RIGHT,")"},

		{OP_X,"x"},
		{OP_Y,"y"},
		{OP_Z,"z"}
	};

	//SymsContainer::SymsContainer(const SymsContainer& other) {
	//	for (auto& it : other.syms_) {
	//		char* p = new char[strlen(it.first) + 1];
	//		strcpy(p, it.first);
	//		syms_.insert(std::make_pair(p, it.second));
	//	}
	//}

	SymsContainer::SymsContainer(SymsContainer&& other) {
		syms_ = std::move(other.syms_);
	}

	SymsContainer& SymsContainer::operator=(SymsContainer&& other) {
		syms_ = std::move(other.syms_);
		return *this;
	}

	SymsContainer::~SymsContainer() {
		clear();
	}

	void SymsContainer::clear() {
		for (auto& it : syms_) {
			delete[] it.first;
		}
		syms_.clear();
	}

	float SymsContainer::getValue(const char* key) {
		char* p = const_cast<char*>(key);
		auto it = syms_.find(p);
		if (it == syms_.end())return 0.0;
		return it->second;
	}

	char* SymsContainer::setValue(const char* data, float value) {
		if (data == nullptr) {
			assert(false);
			return nullptr;
		}
		char* p = const_cast<char*>(data);
		auto iter = syms_.find(p);
		if (iter != syms_.end()) {
			iter->second = value;
			return iter->first;
		}
		else {
			size_t l = strlen(data);
			char* cpy = new char[l + 1];
			strcpy(cpy, data);
			auto pr = syms_.insert(std::make_pair(cpy, value));
			if (pr.second)
				return pr.first->first;
			else
				return nullptr;
		}
		return nullptr;
	}

	AExpression::AExpression()
	{

	}

	AExpression::AExpression(AExpression&& r)
	{
		str_ = std::move(r.str_);
		tokens_ = std::move(r.tokens_);
		prefixTokens_ = std::move(r.prefixTokens_);
		dag_ = std::move(r.dag_);
		symsTable_ = std::move(r.symsTable_);
	}

	AExpression::AExpression(const char* expression) {
		init(expression);
	}

	AExpression AExpression::compose(const AExpression& other) {
		AExpression res;
		return res;//TODO
	}

	void AExpression::setExpression(const char* expression) {
		clear();
		init(expression);
	}

	float AExpression::value(float x, float y, float z) {
		if (dag_.empty()) return 0.f;
		for (int i = 0; i < dag_.size(); ++i) {
			Token& dag = dag_[i];
			switch (dag.code) {
				case OP_FLOAT: 
				{
					dag.value = dag.data.real;
				}
				break;
				case OP_X:
				{
					dag.value = x;
				}
				break;
				case OP_Y:
				{
					dag.value = y;
				}
				break;
				case OP_Z:
				{
					dag.value = z;
				}
				break;
				case OP_SYM:
				{
					dag.value = symsTable_.getValue(dag.data.sym);
				}
				break;
				case OP_ABS:
				{
					dag.value = fabs(dag_[dag.left].value);
				}
				break;
				case OP_SIN:
				{
					dag.value = sin(dag_[dag.left].value);
				}
				break;
				case OP_COS:
				{
					dag.value = cos(dag_[dag.left].value);
				}
				break;
				case OP_ATAN:
				{
					dag.value = atan2(dag_[dag.left].value, dag_[dag.right].value);
				}
				break;
				case OP_SQRT:
				{
					dag.value = sqrt(dag_[dag.left].value);
				}
				break;
				case OP_ADD:
				{
					dag.value = dag_[dag.left].value + dag_[dag.right].value;
				}
				break;
				case OP_SUB:
				{
					dag.value = dag_[dag.left].value - dag_[dag.right].value;
				}
				break;
				case OP_MUL:
				{
					dag.value = dag_[dag.left].value * dag_[dag.right].value;
				}
				break;
				case OP_DIV:
				{
					dag.value = dag_[dag.left].value / dag_[dag.right].value;
				}
				break;
				case OP_POW:
				{
					dag.value = std::pow(dag_[dag.left].value, dag_[dag.right].value);
				}
				break;
				case OP_MIN:
				{
					dag.value = std::min<float>(dag_[dag.left].value, dag_[dag.right].value);
				}
				break;
				case OP_MAX:
				{
					dag.value = std::max<float>(dag_[dag.left].value, dag_[dag.right].value);
				}
				break;
				case OP_CLAD:
				{
					dag.value = dag_[dag.left].value < dag_[dag.right].value ? dag_[dag.right].value : dag_[dag.left].value;
				}
				break;
				case OP_CLAU:
				{
					dag.value = dag_[dag.left].value > dag_[dag.right].value ? dag_[dag.right].value : dag_[dag.left].value;
				}
				break;
				default:
				{
					assert(false);
				}
				break;
			}
		}
		return dag_.back().value;
	}

	void AExpression::setSymValue(const char* sym, float value) {
		symsTable_.setValue(sym,value);
	}

	const std::vector<Token>& AExpression::getDAG() const {
		return dag_;
	}

	std::vector<Token> AExpression::getSymValuedDAG() {
		std::vector<Token> res;
		for (int i = 0; i < dag_.size(); ++i) {
			res.push_back(dag_[i]);
			if(dag_[i].code == OP_SYM){
				res.back().data.real = symsTable_.getValue(dag_[i].data.sym);
			}
		}
		return res;
	}

	bool AExpression::init(const char* expression) {
		str_ = std::string(expression);
		bool ok = lexer(expression);
		if (!ok) return false;

		ok = toPrefix();
		if (!ok) return false;
		printPrefixTokens();

		ok = toDAG();
		printDAG();

		return ok;
	}

	std::string AExpression::oriStr() const {
		return str_;
	}

	//提取tokens
	bool AExpression::lexer(const char* expression) {
		int l = strlen(expression);
		std::string buf;
		for (int i = 0; i < l; ++i) {
			buf.clear();
			char sym = expression[i];
			if (isSpace(sym))
				continue;
			if (sym == ',') {
				continue;
			}
			if (sym == '\t' || sym == '\n') {
				continue;
			}		
			if (isLetter(sym)) {
				while (isLetter(sym)) {
					buf.push_back(sym);
					++i;
					if (i >= l) break;
					sym = expression[i];
				}
				std::cout << buf << std::endl;
				Token token;
				//keyword!
				if (buf == "sqrt") {
					token.code = OP_SQRT;
				}
				else if (buf == "atan") {
					token.code = OP_ATAN;
				}
				else if (buf == "min") {
					token.code = OP_MIN;
				}
				else if (buf == "max") {
					token.code = OP_MAX;
				}
				else if (buf=="abs") {
					token.code = OP_ABS;
				}
				else if (buf == "exp") {
					token.code = OP_EXP;
				}
				else if (buf == "ln") {
					token.code = OP_LN;
				}
				else if (buf == "clad") {
					token.code = OP_CLAD;
				}
				else if (buf == "clau") {
					token.code = OP_CLAU;
				}
				else if (buf == "sin") {
					token.code = OP_SIN;
				}
				else if (buf == "cos") {
					token.code = OP_COS;
				}
				else if (buf == "x") {
					token.code = OP_X;
				}
				else if (buf == "y") {
					token.code = OP_Y;
				}
				else if (buf == "z") {
					token.code = OP_Z;
				}
				else {
					token.code = OP_SYM;
					char* key = symsTable_.setValue(buf.c_str(), 0.f);
					assert(key != nullptr);
					token.data.sym = key;
				}
				tokens_.push_back(token);
				--i;
				continue;
			}
			if (isNum(sym)) {
				while (isNum(sym)) {
					buf.push_back(sym);
					++i;
					if (i >= l) break;
					sym = expression[i];
				}
				if (i < l && expression[i] == '.') {
					buf.push_back('.');
					i += 1;
					while (i < l && isNum(sym = expression[i])) {
						buf.push_back(sym);
						++i;
					}
				}
				//buf to float
				float num = std::stof(buf);
				Token token;
				token.code = OP_FLOAT;
				token.data.real = num;
				tokens_.push_back(token);
				--i;
				continue;
			}
			if (sym == '+') {
				Token token;
				token.code = OP_ADD;
				tokens_.emplace_back(token);
				continue;
			}
			if (sym == '-') {
				Token token;
				token.code = OP_SUB;
				tokens_.emplace_back(token);
				continue;
			}
			if (sym == '*') {
				Token token;
				token.code = OP_MUL;
				tokens_.emplace_back(token);
				continue;
			}
			if (sym == '/') {
				Token token;
				token.code = OP_DIV;
				tokens_.emplace_back(token);
				continue;
			}
			if (sym == '^') {
				Token token;
				token.code = OP_POW;
				tokens_.emplace_back(token);
				continue;
			}
			if (sym == '(') {
				Token token;
				token.code = OP_LEFT;
				tokens_.emplace_back(token);
				continue;
			}
			if (sym == ')') {
				Token token;
				token.code = OP_RIGHT;
				tokens_.emplace_back(token);
				continue;
			}
			
			//if get here (some un expectedthing happen),just remember
			//some info and return false;
			{
				//TODO
				assert(false);
				return false;
			}

		}
		return true;
	}

	//将中缀表达式转为前缀表达式,use the algorithm from
	//https://baike.baidu.com/item/%E5%89%8D%E7%BC%80%E8%A1%A8%E8%BE%BE%E5%BC%8F/5461072?fr=aladdin
	bool AExpression::toPrefix() {
		std::vector<Token> signs;
		std::vector<Token> output;
		for (int i = tokens_.size() - 1; i >= 0; --i) {
			Token& tk = tokens_[i];
			if (isNumber(tk)) {
				output.push_back(tk);
			}
			else if (tk.code == OP_RIGHT) {
				signs.push_back(tk);
			}
			else if (tk.code == OP_LEFT) {
				bool bingo = false;
				for (int k = signs.size() - 1; k >= 0; --k) {
					if (signs[k].code != OP_RIGHT) {
						output.push_back(signs[k]);
					}
					else {
						bingo = true;
						signs.resize(k);
						break;
					}
				}
				if (bingo == false) {
					assert(false);//TODO add info
					return false;
				}
			}
			else{
				if (isBinary(tk)) {
					while (!(signs.empty() || comparePriority(signs.back().code, tk.code) != 1
						|| signs.back().code == OP_LEFT || signs.back().code == OP_RIGHT))
					{
						output.push_back(signs.back());
						signs.pop_back();
					}
					signs.push_back(tk);
				}
				else if (isUnary(tk)) {
					output.push_back(tk);
				}
				else {
					assert(false);
				}
			}
		}
		for (int i = signs.size() - 1; i >= 0; --i) {
			output.push_back(signs[i]);
		}
		std::reverse(output.begin(), output.end());
		std::swap(prefixTokens_, output);
	}

	bool AExpression::toDAG() {
		std::vector<Token> dag;
		std::vector<Token> stack;
		for (int i = prefixTokens_.size() - 1; i >= 0; --i) {
			if (isNumber(prefixTokens_[i])) {
				stack.push_back(prefixTokens_[i]);
			}
			else {
				int n = prefixTokens_[i].inputParamNum();
				assert(stack.size() >= n);
				if (n == 2) {
					Token left = stack.back();
					stack.pop_back();
					Token right = stack.back();
					stack.pop_back();
					//left
					for (int k = 0; k < dag.size(); ++k) {
						if (tokenEqual(left, dag[k])) {
							prefixTokens_[i].left = k;
							break;
						}
					}
					if (prefixTokens_[i].left == -1) {
						dag.push_back(left);
						prefixTokens_[i].left = dag.size() - 1;
					}
					//right
					for (int k = 0; k < dag.size(); ++k) {
						if (tokenEqual(right, dag[k])) {
							prefixTokens_[i].right = k;
							break;
						}
					}
					if (prefixTokens_[i].right == -1) {
						dag.push_back(right);
						prefixTokens_[i].right = dag.size() - 1;
					}
					//self
					stack.push_back(prefixTokens_[i]);
				}
				else if (n == 1) {
					Token child = stack.back();
					stack.pop_back();
					for (int k = 0; k < dag.size(); ++k) {
						if (tokenEqual(child, dag[k])) {
							prefixTokens_[i].left = k;
							break;
						}
					}
					if (prefixTokens_[i].left == -1) {
						dag.push_back(child);
						prefixTokens_[i].left = dag.size() - 1;
					}
					stack.push_back(prefixTokens_[i]);
				}
				else {
					assert(false);
				}
			}
		}
		if (!stack.empty()) {
			//can be more than one?
			assert(stack.size() == 1);
			if (stack.size() != 1) {
				clear();
				return false;
			}
				
			dag.push_back(stack.back());
			stack.pop_back();
		}
		dag_ = std::move(dag);
		return true;
	}

	void AExpression::printPrefixTokens() {
		for (int i = 0; i < prefixTokens_.size(); ++i) {
			printToken(prefixTokens_[i]);
		}
		std::cout << std::endl;
	}

	void AExpression::printDAG() {
		for (int i = dag_.size() - 1; i >= 0; --i) {
			Token& tk = dag_[i];
			std::cout << "index:" << i << std::endl;
			printToken(tk);
			std::cout << ":"<< tk.left << "," << tk.right;
			//std::cout << ":";
			//if (tk.left >= 0) {
			//	printToken(dag_[tk.left]);
			//}
			//else {
			//	std::cout << "-1";
			//}
			//std::cout << ",";
			//if (tk.right >= 0) {
			//	printToken(dag_[tk.right]);
			//}
			//else {
			//	std::cout << "-1";
			//}
			std::cout << std::endl;
			std::cout << std::endl;
		}
	}

	void AExpression::clear() {
		tokens_.clear();

		prefixTokens_.clear();
		dag_.clear();

		symsTable_.clear();
	}

	void AExpression::printToken(const Token& tk) {
		if (tk.code == OP_SYM) {
			std::cout << std::string(tk.data.sym);
		}
		else if (tk.code == OP_FLOAT) {
			std::cout << tk.data.real;
		}
		else if (tk.code == OP_INT) {

		}
		else {
			std::cout << g_signTable[tk.code];
		}
	}
}