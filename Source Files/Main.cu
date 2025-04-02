#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef __INTELLISENSE__
#define __CUDACC__
#endif // __INTELLISENSE__
#include <cooperative_groups.h>
#ifdef __INTELLISENSE__
#undef __CUDACC__
#endif // __INTELLISENSE__

#include "../Header Files/gputimer.h"

using std::cin;
using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::unordered_map;
using std::ifstream;
using std::vector;
using std::milli;
using std::chrono::steady_clock;
using std::chrono::duration;
using std::to_string;
using std::copy;

namespace cg = cooperative_groups;

constexpr int gep = 2; // opening penalty
constexpr int gop = 3; // extend penalty
constexpr int shift = 4; // shift penalty
constexpr int infn = -999;
string myArray[40000][5];

int blosum62mat[24][24];
__device__ __constant__ int d_blosum62mat[24][24];
int score_top[4];

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at: %s : %d\n", file, line);
        fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);;
        exit(1);
    }
}

char DNA_to_Protein(string a) {
    unordered_map<string, char> DP{
        {"TTT", 'F'}, {"TTC", 'F'},
        {"TTA", 'L'}, {"TTG", 'L'},
        {"TCT", 'S'}, {"TCC", 'S'}, {"TCA", 'S'}, {"TCG", 'S'},
        {"TAT", 'Y'}, {"TAC", 'Y'},
        {"TGT", 'C'}, {"TGC", 'C'},
        {"TGG", 'W'},
        {"TAA", 'X'}, {"TAG", 'X'}, {"TGA", 'X'},
        {"CTT", 'L'}, {"CTC", 'L'}, {"CTA", 'L'}, {"CTG", 'L'},
        {"CCT", 'P'}, {"CCC", 'P'}, {"CCA", 'P'}, {"CCG", 'P'},
        {"CAT", 'H'}, {"CAC", 'H'},
        {"CAA", 'Q'}, {"CAG", 'Q'},
        {"CGA", 'R'}, {"CGT", 'R'}, {"CGC", 'R'}, {"CGG", 'R'},
        {"ATT", 'I'}, {"ATC", 'I'}, {"ATA",'I'},
        {"ATG", 'M'},
        {"ACT", 'T'}, {"ACA", 'T'}, {"ACG", 'T'}, {"ACC", 'T'},
        {"AAT", 'N'}, {"AAC", 'N'},
        {"AAG", 'K'}, {"AAA", 'K'},
        {"AGT", 'S'}, {"AGC", 'S'},
        {"AGA", 'R'}, {"AGG", 'R'},
        {"GTT", 'V'}, {"GTA", 'V'}, {"GTG", 'V'}, {"GTC", 'V'},
        {"GCT", 'A'}, {"GCC", 'A'}, {"GCG", 'A'}, {"GCA", 'A'},
        {"GAC", 'D'}, {"GAT", 'D'},
        {"GAA", 'E'}, {"GAG", 'E'},
        {"GGG", 'G'}, {"GGC", 'G'}, {"GGA", 'G'}, {"GGT", 'G'}
    };
    return DP[a];
}

__device__ char d_DNA_to_Protein(const char* dna_seq, int dna_index_1, int dna_index_2, int dna_index_3) {
    char codon[4] = "   ";
    codon[0] = dna_seq[dna_index_1];
    codon[1] = dna_seq[dna_index_2];
    codon[2] = dna_seq[dna_index_3];
    codon[3] = '\0';

    if (codon[0] == 'T') {
        if (codon[1] == 'T') {
            if (codon[2] == 'T' || codon[2] == 'C') return 'F';
            if (codon[2] == 'A' || codon[2] == 'G') return 'L';
        }
        else if (codon[1] == 'C') {
            return 'S';
        }
        else if (codon[1] == 'A') {
            if (codon[2] == 'T' || codon[2] == 'C') return 'Y';
            if (codon[2] == 'A' || codon[2] == 'G') return 'X';
        }
        else if (codon[1] == 'G') {
            if (codon[2] == 'T' || codon[2] == 'C') return 'C';
            if (codon[2] == 'G') return 'W';
            if (codon[2] == 'A') return 'X';
        }
    }
    else if (codon[0] == 'C') {
        if (codon[1] == 'T') return 'L';
        if (codon[1] == 'C') return 'P';
        if (codon[1] == 'A') {
            if (codon[2] == 'T' || codon[2] == 'C') return 'H';
            if (codon[2] == 'A' || codon[2] == 'G') return 'Q';
        }
        else if (codon[1] == 'G') return 'R';
    }
    else if (codon[0] == 'A') {
        if (codon[1] == 'T') {
            if (codon[2] == 'A' || codon[2] == 'C' || codon[2] == 'T') return 'I';
            if (codon[2] == 'G') return 'M';
        }
        if (codon[1] == 'C') return 'T';

        if (codon[1] == 'A') {
            if (codon[2] == 'T' || codon[2] == 'C') return 'N';
            if (codon[2] == 'G' || codon[2] == 'A') return 'K';

        }

        if (codon[1] == 'G') {
            if (codon[2] == 'T' || codon[2] == 'C') return 'S';
            if (codon[2] == 'A' || codon[2] == 'G') return 'R';
        }
    }
    else if (codon[0] == 'G') {
        if (codon[1] == 'T') return 'V';
        if (codon[1] == 'C') return 'A';
        if (codon[1] == 'A') {
            if (codon[2] == 'T' || codon[2] == 'C') return 'D';
            if (codon[2] == 'G' || codon[2] == 'A') return 'E';
        }
        if (codon[1] == 'G') return 'G';
    }
	return ' ';
}

string reverse_complement(string str) {
    unordered_map<char, char> RC{
        {'A', 'T'}, {'C', 'G'}, {'T', 'A'}, {'G', 'C'}
    };

    reverse(str.begin(), str.end());
    for (unsigned int i = 0; i < str.length(); i++) {
        str[i] = RC[str[i]];
    }
    return str;
}

void three_frame(string str, string* frame_one, string* frame_two, string* frame_three) {
    for (unsigned int i = 0; i < str.length(); i += 3) {
        if (i + 2 < str.length()) {
            *frame_one += DNA_to_Protein(str.substr(i, 3));
            *frame_two += DNA_to_Protein(str.substr(i + 1, 3));
            *frame_three += DNA_to_Protein(str.substr(i + 2, 3));
        }
    }
}

int place(char a) {
    unordered_map<char, int> blosumVal{
        {'A', 0}, {'R', 1}, {'N', 2}, {'D', 3}, {'C', 4}, {'Q', 5}, {'E', 6},
        {'G', 7}, {'H', 8}, {'I', 9}, {'L', 10}, {'K', 11 }, {'M', 12},
        {'F', 13}, {'P', 14}, {'S', 15}, {'T', 16}, {'W', 17}, {'Y', 18},
        {'V', 19}, {'B', 20}, {'Z', 21}, {'X', 22}, {'*', 23}
    };
    return blosumVal[(unsigned char)a];
}

__device__ int d_place(char a) {
    if (a == 'A') return 0;
    if (a == 'R') return 1;
    if (a == 'N') return 2;
    if (a == 'D') return 3;
    if (a == 'C') return 4;
    if (a == 'Q') return 5;
    if (a == 'E') return 6;
    if (a == 'G') return 7;
    if (a == 'H') return 8;
    if (a == 'I') return 9;
    if (a == 'L') return 10;
    if (a == 'K') return 11;
    if (a == 'M') return 12;
    if (a == 'F') return 13;
    if (a == 'P') return 14;
    if (a == 'S') return 15;
    if (a == 'T') return 16;
    if (a == 'W') return 17;
    if (a == 'Y') return 18;
    if (a == 'V') return 19;
    if (a == 'B') return 20;
    if (a == 'Z') return 21;
    if (a == 'X') return 22;
    if (a == '*') return 23;
    return -1;
}

int score(char a, char b) {
    int dA, dB;
    dA = place(a);
    dB = place(b);
    return blosum62mat[dA][dB];
}

__device__ int d_score(char a, char b) {
    int dA, dB;
    dA = d_place(a);
    dB = d_place(b);
    return d_blosum62mat[dA][dB];
}

void readBlosum62() {
    ifstream file("./Resource Files/BLOSUM62.txt");
	if (!file.is_open()) {
        cerr << "Error opening the BLOSUM62 file!" << endl;
		exit(1);
	}
    string skip;
    getline(file, skip);
    getline(file, skip);

    char c = NULL;

	for (int i = 0; i < 24; i++) {
		file >> c;
		for (int j = 0; j < 24; j++) {
			file >> blosum62mat[i][j];
		}
	}
	file.close();
}

vector<string> readFastaSequences(const string& filename) {
    ifstream file("./Resource Files/" + filename + ".fasta");
    vector<string> sequences;
    string line, sequence;

    if (!file) {
        cerr << "Error: Unable to open file " << filename << endl;
        return sequences;
    }

    while (getline(file, line)) {
        if (line.empty()) continue;
        if (line[0] == '>') {
            if (!sequence.empty()) {
                sequences.push_back(sequence);
                sequence.clear();
            }
        }
        else {
            sequence += line;
        }
    }

    if (!sequence.empty()) {
        sequences.push_back(sequence);
    }

    file.close();

    return sequences;
}

vector<string> readFastaIDs(const string& filename) {
    ifstream file("./Resource Files/" + filename + ".fasta");
    vector<string> sequenceIDs;
    string line;

    if (!file) {
        cerr << "Error: Unable to open file " << filename << endl;
        return sequenceIDs;
    }

    while (getline(file, line)) {
        if (!line.empty() && line[0] == '>') {
            sequenceIDs.push_back(line.substr(1)); // Remove '>' and store only the ID
        }
    }

    return sequenceIDs;
}

void init_local_v2(string input_seq, string ref_seq, int** sc_mat, int** ins_mat, int** del_mat, int** t_sc_mat, int** t_ins_mat, int** t_del_mat) {
    size_t N = input_seq.length();
    size_t M = ref_seq.length() + 1;

    for (size_t i = 0; i < N; i++) {
        ins_mat[i][0] = infn;
        t_ins_mat[i][0] = infn;
    }

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            if (i == 0 || j == 0) {
                sc_mat[i][j] = 0;
            }
        }
    }

    for (size_t j = 0; j < M; j++) {
        del_mat[0][j] = infn;
        del_mat[2][j] = infn;
        del_mat[3][j] = infn;
        del_mat[1][j] = sc_mat[0][j] - gop - gep;

        t_del_mat[0][j] = infn;
        t_del_mat[2][j] = infn;
        t_del_mat[3][j] = infn;
        t_del_mat[1][j] = 1;
    }

    int insert = 0;
    int del = 0;
    int xscore = 0;
    int end = 3;

    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 1; j < M; j++) {
            insert = ins_mat[i][j - 1] - gep;
            xscore = sc_mat[i][j - 1] - gop - gep;
            if (insert > xscore) {
                ins_mat[i][j] = insert;
            }
            else {
                ins_mat[i][j] = xscore;
            }

            insert = ins_mat[i][j];
            del = del_mat[i][j];

            if (i == 1) {
                xscore = sc_mat[0][j - 1] + score(DNA_to_Protein(input_seq.substr(i - 1, end)), ref_seq[j - 1]);
                if (insert >= del && insert >= xscore) {
                    sc_mat[i][j] = insert;
                }
                else if (del >= insert && del >= xscore) {
                    sc_mat[i][j] = del;
                }
                else {
                    sc_mat[i][j] = xscore;
                }

                if (sc_mat[i][j] == ins_mat[i][j]) {
                    t_sc_mat[i][j] = -2;
                }
                else if (sc_mat[i][j] == del_mat[i][j]) {
                    t_sc_mat[i][j] = -1;
                }
                else if (sc_mat[i][j] == xscore) {
                    t_sc_mat[i][j] = 1;
                }
            }
            else if (i == 2) {
                xscore = sc_mat[0][j - 1] + score(DNA_to_Protein(input_seq.substr(i - 1, end)), ref_seq[j - 1]) - shift;
                if (insert >= xscore) {
                    sc_mat[i][j] = insert;
                }
                else {
                    sc_mat[i][j] = xscore;
                }

                if (sc_mat[i][j] == ins_mat[i][j]) {
                    t_sc_mat[i][j] = -2;
                }
                else if (sc_mat[i][j] == xscore) {
                    t_sc_mat[i][j] = 2;
                }
            }
            else if (i == 3) {
                xscore = sc_mat[1][j - 1] + score(DNA_to_Protein(input_seq.substr(i - 1, end)), ref_seq[j - 1]) - shift;
                if (insert >= xscore) {
                    sc_mat[i][j] = insert;
                }
                else {
                    sc_mat[i][j] = xscore;
                }

                if (sc_mat[i][j] == ins_mat[i][j]) {
                    t_sc_mat[i][j] = -2;
                }
                else if (sc_mat[i][j] == xscore) {
                    t_sc_mat[i][j] = 2;
                }
            }

            if (sc_mat[i][j] < 0)
                sc_mat[i][j] = 0;
        }
    }
}

void init_local_v2_cuda(string input_seq, string ref_seq, int* u_sc_mat, int* u_ins_mat, int* u_del_mat, int* u_t_sc_mat, int* u_t_ins_mat, int* u_t_del_mat, size_t N, size_t M) {
    for (size_t i = 0; i < N; i++) {
        u_ins_mat[i * M] = infn;
        u_t_ins_mat[i * M] = infn;
    }

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            if (i == 0 || j == 0) {
                u_sc_mat[i * M + j] = 0;
            }
        }
    }

    for (size_t j = 0; j < M; j++) {
        u_del_mat[0 * M + j] = infn;
        u_del_mat[2 * M + j] = infn;
        u_del_mat[3 * M + j] = infn;
        u_del_mat[1 * M + j] = u_sc_mat[0 * M + j] - gop - gep;

        u_t_del_mat[0 * M + j] = infn;
        u_t_del_mat[2 * M + j] = infn;
        u_t_del_mat[3 * M + j] = infn;
        u_t_del_mat[1 * M + j] = 1;
    }

    int insert = 0;
    int del = 0;
    int xscore = 0;
    int end = 3;

    for (int i = 0; i < 4; i++) {
        for (int j = 1; j < M; j++) {
            insert = u_ins_mat[i * M + (j - 1)] - gep;
            xscore = u_sc_mat[i * M + (j - 1)] - gop - gep;

            if (insert > xscore) {
                u_ins_mat[i * M + j] = insert;
            }
            else {
                u_ins_mat[i * M + j] = xscore;
            }

            insert = u_ins_mat[i * M + j];
            del = u_del_mat[i * M + j];

            if (i == 1) {
                xscore = u_sc_mat[0 * M + (j - 1)] + score(DNA_to_Protein(input_seq.substr(i - 1, end)), ref_seq[j - 1]);
                if (insert >= del && insert >= xscore) {
                    u_sc_mat[i * M + j] = insert;
                }
                else if (del >= insert && del >= xscore) {
                    u_sc_mat[i * M + j] = del;
                }
                else {
                    u_sc_mat[i * M + j] = xscore;
                }

                if (u_sc_mat[i * M + j] == u_ins_mat[i * M + j]) {
                    u_t_sc_mat[i * M + j] = -2;
                }
                else if (u_sc_mat[i * M + j] == u_del_mat[i * M + j]) {
                    u_t_sc_mat[i * M + j] = -1;
                }
                else if (u_sc_mat[i * M + j] == xscore) {
                    u_t_sc_mat[i * M + j] = 1;
                }
            }
            else if (i == 2) {
                xscore = u_sc_mat[0 * M + (j - 1)] + score(DNA_to_Protein(input_seq.substr(i - 1, end)), ref_seq[j - 1]) - shift;
                if (insert >= xscore) {
                    u_sc_mat[i * M + j] = insert;
                }
                else {
                    u_sc_mat[i * M + j] = xscore;
                }

                if (u_sc_mat[i * M + j] == u_ins_mat[i * M + j]) {
                    u_t_sc_mat[i * M + j] = -2;
                }
                else if (u_sc_mat[i * M + j] == xscore) {
                    u_t_sc_mat[i * M + j] = 2;
                }
            }
            else if (i == 3) {
                xscore = u_sc_mat[1 * M + (j - 1)] + score(DNA_to_Protein(input_seq.substr(i - 1, end)), ref_seq[j - 1]) - shift;
                if (insert >= xscore) {
                    u_sc_mat[i * M + j] = insert;
                }
                else {
                    u_sc_mat[i * M + j] = xscore;
                }

                if (u_sc_mat[i * M + j] == u_ins_mat[i * M + j]) {
                    u_t_sc_mat[i * M + j] = -2;
                }
                else if (u_sc_mat[i * M + j] == xscore) {
                    u_t_sc_mat[i * M + j] = 2;
                }
            }

            if (u_sc_mat[i * M + j] < 0) {
                u_sc_mat[i * M + j] = 0;
            }
        }
    }
}

void scoring_local_v2(string input_seq, string ref_seq, int** sc_mat, int** ins_mat, int** del_mat, int** t_sc_mat, int** t_ins_mat, int** t_del_mat) {
    size_t N = input_seq.length();
    size_t M = ref_seq.length() + 1;
    int insert = 0;
    int del = 0;
    int xscore = 0;
    int end = 3;
    int sc_1 = 0, sc_2 = 0, sc_3 = 0;
    int scoring = 0;
    char prot_seq;

    for (size_t i = 4; i < N; i++) {
        for (size_t j = 1; j < M; j++) {
            prot_seq = DNA_to_Protein(input_seq.substr(i - 1, end));
            scoring = score(prot_seq, ref_seq[j - 1]);
            insert = ins_mat[i][j - 1] - gep;
            xscore = sc_mat[i][j - 1] - gop - gep;
            if (insert > xscore) {
                ins_mat[i][j] = insert;
            }
            else {
                ins_mat[i][j] = xscore;
            }

            if (ins_mat[i][j] == insert) {
                t_ins_mat[i][j] = 0;
            }
            else if (ins_mat[i][j] == xscore) {
                t_ins_mat[i][j] = 1;
            }

            del = del_mat[i - 3][j] - gep;
            xscore = sc_mat[i - 3][j] - gop - gep;

            if (del >= xscore) {
                del_mat[i][j] = del;
            }
            else {
                del_mat[i][j] = xscore;
            }

            if (del_mat[i][j] == del) {
                t_del_mat[i][j] = 0;
            }
            else if (del_mat[i][j] == xscore) {
                t_del_mat[i][j] = 1;
            }

            if (i < N - 1) {
                insert = ins_mat[i][j];
                del = del_mat[i][j];
                sc_1 = sc_mat[i - 2][j - 1] + scoring - shift;
                sc_2 = sc_mat[i - 3][j - 1] + scoring;
                sc_3 = sc_mat[i - 4][j - 1] + scoring - shift;

                if (insert >= del && insert >= sc_1 && insert >= sc_2 && insert >= sc_3) {
                    sc_mat[i][j] = insert;
                }
                else if (del >= insert && del >= sc_1 && del >= sc_2 && del >= sc_3) {
                    sc_mat[i][j] = del;
                }
                else if (sc_1 >= insert && sc_1 >= del && sc_1 >= sc_2 && sc_1 >= sc_3) {
                    sc_mat[i][j] = sc_1;
                }
                else if (sc_2 >= insert && sc_2 >= del && sc_2 >= sc_1 && sc_2 >= sc_3) {
                    sc_mat[i][j] = sc_2;
                }
                else if (sc_3 >= insert && sc_3 >= del && sc_3 >= sc_1 && sc_3 >= sc_2) {
                    sc_mat[i][j] = sc_3;
                }

                if (sc_mat[i][j] == insert) {
                    t_sc_mat[i][j] = -2;
                }
                else if (sc_mat[i][j] == del) {
                    t_sc_mat[i][j] = -1;
                }
                else if (sc_mat[i][j] == sc_1) {
                    t_sc_mat[i][j] = 2;
                }
                else if (sc_mat[i][j] == sc_2) {
                    t_sc_mat[i][j] = 3;
                }
                else if (sc_mat[i][j] == sc_3) {
                    t_sc_mat[i][j] = 4;
                }
            }

            if (sc_mat[i][j] < 0) {
                sc_mat[i][j] = 0;
            }
        }
    }
    for (size_t i = N - 1; i < N; i++) {
        for (size_t j = 1; j < M; j++) {
            insert = ins_mat[i][j - 1] - gep;
            xscore = sc_mat[i][j - 1] - gop - gep;
            if (insert >= xscore) {
                ins_mat[i][j] = insert;
            }
            else {
                ins_mat[i][j] = xscore;
            }

            sc_mat[i][j] = infn;
            t_sc_mat[i][j] = infn;
        }
    }
}

__global__ void scoring_local_v2_cuda(const char* input_seq, const char* ref_seq, int* u_sc_mat, int* u_ins_mat, int* u_del_mat, int* u_t_sc_mat, int* u_t_ins_mat, int* u_t_del_mat, unsigned int N, unsigned int M, unsigned int submatrixStartX, unsigned int submatrixStartY, unsigned int submatrixSide) {
    cg::thread_block block = cg::this_thread_block();


    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    i += submatrixStartY;
    j += submatrixStartX;

    int insert = 0;
    int del = 0;
    int xscore = 0;
    int sc_1 = 0, sc_2 = 0, sc_3 = 0;
    int scoring = 0;
    char prot_seq;

    if (i >= 4 && i < N && j >= 1 && j < M) {
        for (unsigned int diag = 0; diag < 2 * submatrixSide - 1; ++diag) {
            if (i < submatrixStartY + submatrixSide && j < submatrixStartX + submatrixSide && (i - submatrixStartY) + (j - submatrixStartX) == diag) {
                prot_seq = d_DNA_to_Protein(input_seq, i - 1, i, i + 1);
                scoring = d_score(prot_seq, ref_seq[j - 1]);
                insert = u_ins_mat[i * M + (j - 1)] - gep;
                xscore = u_sc_mat[i * M + (j - 1)] - gop - gep;

                if (insert > xscore) {
                    u_ins_mat[i * M + j] = insert;
                }
                else {
                    u_ins_mat[i * M + j] = xscore;
                }

                if (u_ins_mat[i * M + j] == insert) {
                    u_t_ins_mat[i * M + j] = 0;
                }
                else if (u_ins_mat[i * M + j] == xscore) {
                    u_t_ins_mat[i * M + j] = 1;
                }

                del = u_del_mat[(i - 3) * M + j] - gep;
                xscore = u_sc_mat[(i - 3) * M + j] - gop - gep;

                if (del >= xscore) {
                    u_del_mat[i * M + j] = del;
                }
                else {
                    u_del_mat[i * M + j] = xscore;
                }

                if (u_del_mat[i * M + j] == del) {
                    u_t_del_mat[i * M + j] = 0;
                }
                else if (u_del_mat[i * M + j] == xscore) {
                    u_t_del_mat[i * M + j] = 1;
                }

                if (i < N - 1) {
                    insert = u_ins_mat[i * M + j];
                    del = u_del_mat[i * M + j];
                    sc_1 = u_sc_mat[(i - 2) * M + (j - 1)] + scoring - shift;
                    sc_2 = u_sc_mat[(i - 3) * M + (j - 1)] + scoring;
                    sc_3 = u_sc_mat[(i - 4) * M + (j - 1)] + scoring - shift;

                    if (insert >= del && insert >= sc_1 && insert >= sc_2 && insert >= sc_3) {
                        u_sc_mat[i * M + j] = insert;
                    }
                    else if (del >= insert && del >= sc_1 && del >= sc_2 && del >= sc_3) {
                        u_sc_mat[i * M + j] = del;
                    }
                    else if (sc_1 >= insert && sc_1 >= del && sc_1 >= sc_2 && sc_1 >= sc_3) {
                        u_sc_mat[i * M + j] = sc_1;
                    }
                    else if (sc_2 >= insert && sc_2 >= del && sc_2 >= sc_1 && sc_2 >= sc_3) {
                        u_sc_mat[i * M + j] = sc_2;
                    }
                    else if (sc_3 >= insert && sc_3 >= del && sc_3 >= sc_1 && sc_3 >= sc_2) {
                        u_sc_mat[i * M + j] = sc_3;
                    }

                    if (u_sc_mat[i * M + j] == insert) {
                        u_t_sc_mat[i * M + j] = -2;
                    }
                    else if (u_sc_mat[i * M + j] == del) {
                        u_t_sc_mat[i * M + j] = -1;
                    }
                    else if (u_sc_mat[i * M + j] == sc_1) {
                        u_t_sc_mat[i * M + j] = 2;
                    }
                    else if (u_sc_mat[i * M + j] == sc_2) {
                        u_t_sc_mat[i * M + j] = 3;
                    }
                    else if (u_sc_mat[i * M + j] == sc_3) {
                        u_t_sc_mat[i * M + j] = 4;
                    }
                }

                if (u_sc_mat[i * M + j] < 0) {
                    u_sc_mat[i * M + j] = 0;
                }
            }
            cg::sync(block);
        }
    }
    else if (i == N - 1 && j >= 1 && j < M) {
        for (unsigned int diag = 0; diag < 2 * submatrixSide - 1; ++diag) {
            if (i < submatrixStartY + submatrixSide && j < submatrixStartX + submatrixSide && (i - submatrixStartY) + (j - submatrixStartX) == diag) {
                insert = u_ins_mat[i * M + (j - 1)] - gep;
                xscore = u_sc_mat[i * M + (j - 1)] - gop - gep;
                if (insert >= xscore) {
                    u_ins_mat[i * M + j] = insert;
                }
                else {
                    u_ins_mat[i * M + j] = xscore;
                }

                u_sc_mat[i * M + j] = infn;
                u_t_sc_mat[i * M + j] = infn;
            }
        }
    }
}

void top5(int score, int index, int top_i, int top_j, int* score_top, int* top_i_max, int* top_j_max, int* top_indices) {
    // Check if the new score belongs in the top 5
    for (int x = 0; x < 5; x++) {
        if (score >= score_top[x]) {
            for (int y = 4; y > x; y--) {
                score_top[y] = score_top[y - 1];
                top_i_max[y] = top_i_max[y - 1];
                top_j_max[y] = top_j_max[y - 1];
                top_indices[y] = top_indices[y - 1];
            }

            score_top[x] = score;
            top_i_max[x] = top_i;
            top_j_max[x] = top_j;
            top_indices[x] = index;

            break;
        }
    }
}

void routine(int trace, int& i, int& j, string str, string ref_seq, string& final_seq1, string& final_seq2, string& frameshift) {
    int k = 0;
    if (trace == -2) {
        j--;
        k = i + 3;
        final_seq1 += "-";
        final_seq2 += ref_seq[j];
        frameshift += " ";
    }
    else if (trace == -1) {
        i -= 3;
        k = i + 3;
        final_seq1 += DNA_to_Protein(str.substr(k - 1, 3));;
        final_seq2 += "-";
        frameshift += " ";
    }
    else if (trace == 3) {
        i -= 3;
        j--;
        k = i + 3;
        final_seq1 += DNA_to_Protein(str.substr(k - 1, 3));
        final_seq2 += ref_seq[j];
        frameshift += " ";

    }
    else if (trace == 4) {
        i -= 4;
        j--;
        k = i + 4;
        final_seq1 += DNA_to_Protein(str.substr(k - 1, 3));
        final_seq2 += ref_seq[j];
        frameshift += "*";

    }
    else if (trace == 2) {
        i -= 2;
        j--;
        k = i + 2;
        final_seq1 += DNA_to_Protein(str.substr(k - 1, 3));
        final_seq2 += ref_seq[j];
        frameshift += "*";
    }
    else if (trace == 1) {
        i--;
        j--;
        k = i + 1;
        final_seq1 += DNA_to_Protein(str.substr(k - 1, 3));
        final_seq2 += ref_seq[j];
        frameshift += " ";
    }
}

//To be Replaced
void check_index(string input_dna, string ref_prot, string seq_dna, string seq_prot) {
    int i, high = -1, low = -1, hold = 0, temp = -1, curr = 0;
    for (i = 0; i < ref_prot.length(); i++) {
        if (ref_prot[i] == seq_prot[curr]) {
            if (curr == 0) {
                temp = i;
            }
            curr++;

            if (curr > hold) {
                hold = curr;
                low = temp;
                high = i;
            }
        }
        else {
            curr = 0;
            temp = -1;
        }

    }
    if (low != -1 && high != -1)
        cout << "Protein sequence indexes: " << low << " to " << high << endl;
}

void traceV2(string input_seq, string ref_seq, int** sc_mat, int** t_sc_mat, size_t N, size_t M, int index, int* indeces) {
    N = (int)N;
    M = (int)M;
    //indeces[3];
    int i_max = 0, j_max = 0, i = 0, j = 0, max_score = 0, curr_score;
    string f1, f2, f3, seq_dna, seq_prot, frameshift;
    three_frame(input_seq, &f1, &f2, &f3);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            curr_score = sc_mat[i][j];
            if (curr_score > max_score) {
                max_score = curr_score;
                i_max = i;
                j_max = j;
            }
        }
    }
    i = i_max;
    j = j_max;

    indeces[0] = max_score;
    indeces[1] = i_max;
    indeces[2] = j_max;


    while (sc_mat[i][j] != 0) {
        myArray[index][1] = to_string(j);
        routine(t_sc_mat[i][j], i, j, input_seq, ref_seq, seq_dna, seq_prot, frameshift);
    }


    cout << endl;
    reverse(seq_dna.begin(), seq_dna.end());
    reverse(seq_prot.begin(), seq_prot.end());
    reverse(frameshift.begin(), frameshift.end());

    cout << "frame 1: \t";
    for (i = 0; i < f1.length(); i++) {
        cout << f1[i] << "  ";
    }
    cout << endl;
    cout << "frame 2: \t ";
    for (i = 0; i < f2.length(); i++) {
        cout << f2[i] << "  ";
    }
    cout << endl;
    cout << "frame 3: \t  ";
    for (i = 0; i < f3.length(); i++) {
        cout << f3[i] << "  ";
    }
    cout << endl;

    cout << "Output DNA: \t";
    for (i = 0; i < seq_dna.length(); i++) {
        cout << seq_dna[i] << "  ";
    }
    cout << endl;
    cout << "Frameshift: \t";
    for (i = 0; i < frameshift.length(); i++) {
        cout << frameshift[i] << "  ";
    }
    cout << endl;
    cout << "Output Prot: \t";
    for (i = 0; i < seq_prot.length(); i++) {
        cout << seq_prot[i] << "  ";
    }
    cout << endl;
    cout << "Reference Seq: " << ref_seq << endl;
}

void traceV2_check(string input_seq, string ref_seq, int** sc_mat, int** t_sc_mat, size_t N, size_t M, int index, int* indeces) {
    N = (int)N;
    M = (int)M;
    int i_max = 0, j_max = 0, i = 0, j = 0, max_score = 0, curr_score;
    string f1, f2, f3, seq_dna, seq_prot, frameshift;
    three_frame(input_seq, &f1, &f2, &f3);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            curr_score = sc_mat[i][j];
            if (curr_score > max_score) {
                max_score = curr_score;
                i_max = i;
                j_max = j;
            }
        }
    }
    i = i_max;
    j = j_max;

    indeces[0] = max_score;
    indeces[1] = i_max;
    indeces[2] = j_max;

    myArray[index][0] = to_string(max_score);
    myArray[index][2] = to_string(j_max);

}

void traceV2_1d(string input_seq, string ref_seq, int* sc_mat, int* t_sc_mat, int N, int M, int index, int* indeces) {
    int i_max = 0, j_max = 0, i = 0, j = 0, max_score = 0, curr_score;
    string f1, f2, f3, seq_dna, seq_prot, frameshift;
    indeces[3];
    three_frame(input_seq, &f1, &f2, &f3);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            curr_score = sc_mat[i * M + j];
            if (curr_score > max_score) {
                max_score = curr_score;
                i_max = i;
                j_max = j;
            }
        }
    }

    i = i_max;
    j = j_max;

    indeces[0] = max_score;
    indeces[1] = i_max;
    indeces[2] = j_max;

    while (sc_mat[i * M + j] != 0) {

        int score_tc;
        score_tc = t_sc_mat[i * M + j];
        myArray[index][1] = to_string(j);
        routine(t_sc_mat[i * M + j], i, j, input_seq, ref_seq, seq_dna, seq_prot, frameshift);
    }

    cout << endl;
    reverse(seq_dna.begin(), seq_dna.end());
    reverse(seq_prot.begin(), seq_prot.end());
    reverse(frameshift.begin(), frameshift.end());

    cout << "frame 1: \t";
    for (i = 0; i < f1.length(); i++) {
        cout << f1[i] << "  ";
    }
    cout << endl;
    cout << "frame 2: \t ";
    for (i = 0; i < f2.length(); i++) {
        cout << f2[i] << "  ";
    }
    cout << endl;
    cout << "frame 3: \t  ";
    for (i = 0; i < f3.length(); i++) {
        cout << f3[i] << "  ";
    }
    cout << endl;

    cout << "Output DNA: \t";
    for (i = 0; i < seq_dna.length(); i++) {
        cout << seq_dna[i] << "  ";
    }
    cout << endl;
    cout << "Frameshift: \t";
    for (i = 0; i < frameshift.length(); i++) {
        cout << frameshift[i] << "  ";
    }
    cout << endl;
    cout << "Output Prot: \t";
    for (i = 0; i < seq_prot.length(); i++) {
        cout << seq_prot[i] << "  ";
    }
    cout << endl;
    cout << "Reference Seq: " << ref_seq << endl;
}

void traceV2_1d_check(string input_seq, string ref_seq, int* sc_mat, int* t_sc_mat, size_t N, size_t M, int index, int* indeces) {
    N = (int)N;
    M = (int)M;
    int i_max = 0, j_max = 0, i = 0, j = 0, max_score = 0, curr_score;
    string f1, f2, f3, seq_dna, seq_prot, frameshift;
    three_frame(input_seq, &f1, &f2, &f3);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            curr_score = sc_mat[i * M + j];
            if (curr_score > max_score) {
                max_score = curr_score;
                i_max = i;
                j_max = j;
            }
        }
    }
    i = i_max;
    j = j_max;

    indeces[0] = max_score;
    indeces[1] = i_max;
    indeces[2] = j_max;

    myArray[index][0] = to_string(max_score);
    myArray[index][2] = to_string(j_max);

}


int main()
{
    int mode, frame, top, device;
    vector<string> dnaInputs, proteinInputs, proteinIdInputs;
    string protein_sequence, DNA_sequence, DNA_sequence_r, file_name;
    cudaDeviceProp prop;
    GpuTimer timer;

    checkCudaErrors(cudaGetDevice(&device));
    checkCudaErrors(cudaSetDevice(device));

    checkCudaErrors(cudaGetDeviceProperties_v2(&prop, device));

    readBlosum62();

    cudaMemcpyToSymbol(d_blosum62mat, blosum62mat, sizeof(blosum62mat));

    do {
        file_name.clear();
        cout << "Input file name for protein(exclude .fasta/.fastq):" << endl << "--> ";
        cin >> file_name;

        proteinInputs = readFastaSequences(file_name);
        proteinIdInputs = readFastaIDs(file_name);
        if (!proteinInputs.empty() && !proteinIdInputs.empty()) {
            break;
        }
    } while (true);

    do {
        file_name.clear();
        cout << "Input file name for DNA (exclude .fasta/.fastq):" << endl << "--> ";
        cin >> file_name;

        dnaInputs = readFastaSequences(file_name);
        if (!dnaInputs.empty()) {
            break;
        }
    } while (true);

    do {
        cout << "Choose mode [0 - Sequential] [1 - CUDA]:" << endl << "--> ";
        cin >> mode;

        if (mode == 0 || mode == 1)
            break;

        cout << "Invalid input please try again." << endl;
    } while (true);

    do {
        cout << "Choose frame count [3 - 3 Frame] [6 - 6 Frame]:" << endl << "--> ";
        cin >> frame;

        if (frame == 3 || frame == 6) 
            break;
        
        cout << "Invalid input please try again." << endl;
    } while (true);

    do {
        cout << "Show Score [0 - Top 1] [1 - Top 5]:" << endl << "--> ";
        cin >> top;

        if (top == 0 || top == 1)
            break;

        cout << "Invalid input please try again." << endl;

    } while (true);

    for (int index_dna = 0; index_dna < dnaInputs.size(); index_dna++) {

        int* score_top1 = new int [5] {};
        int* top_scores = new int[5] {};
        int* top_i = new int[5] {};
        int* top_j = new int[5] {};
        int* top_indeces = new int[5] {};
        int* index = new int[4] {};
        int top_hold = 0;

        for (int index_prot = 0; index_prot < proteinInputs.size(); index_prot++) {

			DNA_sequence = dnaInputs[index_dna];
			protein_sequence = proteinInputs[index_prot];
            DNA_sequence_r = reverse_complement(DNA_sequence);

			size_t N = DNA_sequence.length();
			size_t M = protein_sequence.length() + 1;

            size_t N_size = N * sizeof(char);
            size_t M_size = M * sizeof(char);
			size_t size = N * M * sizeof(int);
            

            if (mode == 0) {

                int** sc_mat = new int* [N];
                int** ins_mat = new int* [N];
                int** del_mat = new int* [N];
                int** sc_mat_hold = new int* [N];

                int** t_sc_mat = new int* [N];
                int** t_ins_mat = new int* [N];
                int** t_del_mat = new int* [N];
                int** t_sc_mat_hold = new int* [N];

                for (int i = 0; i < N; i++) {
                    sc_mat[i] = new int[M]();
                    ins_mat[i] = new int[M]();
                    del_mat[i] = new int[M]();
                    sc_mat_hold[i] = new int[M]();

                    t_sc_mat[i] = new int[M]();
                    t_ins_mat[i] = new int[M]();
                    t_del_mat[i] = new int[M]();
                    t_sc_mat_hold[i] = new int[M]();
                }
                
				init_local_v2(DNA_sequence, protein_sequence, sc_mat, ins_mat, del_mat, t_sc_mat, t_ins_mat, t_del_mat);

                auto start = steady_clock::now();
				scoring_local_v2(DNA_sequence, protein_sequence, sc_mat, ins_mat, del_mat, t_sc_mat, t_ins_mat, t_del_mat);
                auto end = steady_clock::now();
                auto diff = end - start;

                traceV2_check(DNA_sequence, protein_sequence, sc_mat, t_sc_mat, N, M, index_prot, index);
                top5(index[0], index_prot, index[1], index[3], top_scores, top_i, top_j, top_indeces);
                cout << "Run DNA: " << index_dna << " Prot: " << index_prot << endl << "Time in ms: " << duration<double, milli>(diff).count() << endl;
                    
				if (frame == 6) {
					init_local_v2(DNA_sequence_r, protein_sequence, sc_mat, ins_mat, del_mat, t_sc_mat, t_ins_mat, t_del_mat);

                    auto start = steady_clock::now();
					scoring_local_v2(DNA_sequence_r, protein_sequence, sc_mat, ins_mat, del_mat, t_sc_mat, t_ins_mat, t_del_mat);
                    auto end = steady_clock::now();
                    auto diff = end - start;

                    traceV2_check(DNA_sequence_r, protein_sequence, sc_mat, t_sc_mat, N, M, index_prot, index);
                    top5(index[0], index_prot, index[1], index[3], top_scores, top_i, top_j, top_indeces);
                    cout << "Run DNA: " << index_dna << " Prot: " << index_prot << endl << "Time in ms: " << duration<double, milli>(diff).count() << endl;
				}

                for (int i = 0; i < N_size; i++) {
                    delete[] sc_mat[i];
                    delete[] ins_mat[i];
                    delete[] del_mat[i];
                    delete[] t_sc_mat[i];
                    delete[] t_ins_mat[i];
                    delete[] t_del_mat[i];
                    delete[] sc_mat_hold[i];
                    delete[] t_sc_mat_hold[i];
                }

                delete[] sc_mat;
                delete[] ins_mat;
                delete[] del_mat;
                delete[] t_sc_mat;
                delete[] t_ins_mat;
                delete[] t_del_mat;
                delete[] sc_mat_hold;
                delete[] t_sc_mat_hold;
            }
            else if (mode == 1) {

                char* d_DNA_sequence;
                char* d_protein_sequence;
                char* d_DNA_sequence_r;

                int* u_sc_mat;
                int* u_ins_mat;
                int* u_del_mat;

                int* u_t_sc_mat;
                int* u_t_ins_mat;
                int* u_t_del_mat;

                checkCudaErrors(cudaMalloc(&d_DNA_sequence, N_size));
                checkCudaErrors(cudaMalloc(&d_protein_sequence, M_size));
                checkCudaErrors(cudaMalloc(&d_DNA_sequence_r, N_size));

                checkCudaErrors(cudaMemcpy(d_DNA_sequence, DNA_sequence.c_str(), N_size, cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemcpy(d_protein_sequence, protein_sequence.c_str(), M_size, cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemcpy(d_DNA_sequence_r, DNA_sequence_r.c_str(), N_size, cudaMemcpyHostToDevice));

                checkCudaErrors(cudaMallocManaged(&u_sc_mat, size));
                checkCudaErrors(cudaMallocManaged(&u_ins_mat, size));
                checkCudaErrors(cudaMallocManaged(&u_del_mat, size));

                checkCudaErrors(cudaMallocManaged(&u_t_sc_mat, size));
                checkCudaErrors(cudaMallocManaged(&u_t_ins_mat, size));
                checkCudaErrors(cudaMallocManaged(&u_t_del_mat, size));

                checkCudaErrors(cudaMemset(u_sc_mat, 0, size));
                checkCudaErrors(cudaMemset(u_ins_mat, 0, size));
                checkCudaErrors(cudaMemset(u_del_mat, 0, size));

                checkCudaErrors(cudaMemset(u_t_sc_mat, 0, size));
                checkCudaErrors(cudaMemset(u_t_ins_mat, 0, size));
                checkCudaErrors(cudaMemset(u_t_del_mat, 0, size));

                dim3 blockDimMain(32, 32);
                dim3 gridDimMain(1);
				dim3 blockDimLastRow(1024);
                dim3 gridDimLastRow(((unsigned int)(M - 1) + blockDimLastRow.x - 1) / blockDimLastRow.x);


				unsigned int submatrixSide = blockDimMain.x;
				unsigned int numSubmatrixRows = ((unsigned int)N + submatrixSide - 1) / submatrixSide;
				unsigned int numSubmatrixCols = ((unsigned int)M + submatrixSide - 1) / submatrixSide;

				init_local_v2_cuda(DNA_sequence, protein_sequence, u_sc_mat, u_ins_mat, u_del_mat, u_t_sc_mat, u_t_ins_mat, u_t_del_mat, N, M);

                timer.Start();
                for (unsigned int diag = 0; diag < numSubmatrixRows + numSubmatrixCols - 1; ++diag) {
                    for (unsigned int submatrixY = std::max(0, (int)diag - (int)(numSubmatrixCols - 1)); submatrixY <= diag && submatrixY < numSubmatrixRows; ++submatrixY) {
                        int submatrixX = diag - submatrixY;
                        scoring_local_v2_cuda << <gridDimMain, blockDimMain >> > (d_DNA_sequence, d_protein_sequence, u_sc_mat, u_ins_mat, u_del_mat, u_t_sc_mat, u_t_ins_mat, u_t_del_mat, N, M, submatrixX * submatrixSide, submatrixY * submatrixSide, submatrixSide);
                        checkCudaErrors(cudaGetLastError());
                    }
                    checkCudaErrors(cudaDeviceSynchronize());
                }
                timer.Stop();        

                traceV2_1d_check(DNA_sequence, protein_sequence, u_sc_mat, u_t_sc_mat, N, M, index_prot, index);
                cout << "Run DNA: " << index_dna << " Prot: " << index_prot << endl << "Time in ms: " << timer.Elapsed() << endl;
                top5(index[0], index_prot, index[1], index[2], top_scores, top_i, top_j, top_indeces);
   
				if (frame == 6) {
					init_local_v2_cuda(DNA_sequence_r, protein_sequence, u_sc_mat, u_ins_mat, u_del_mat, u_t_sc_mat, u_t_ins_mat, u_t_del_mat, N, M);
						
                    timer.Start();
                    for (unsigned int diag = 0; diag < numSubmatrixRows + numSubmatrixCols - 1; ++diag) {
                        for (unsigned int submatrixY = std::max(0, (int)diag - (int)(numSubmatrixCols - 1)); submatrixY <= diag && submatrixY < numSubmatrixRows; ++submatrixY) {
                            int submatrixX = diag - submatrixY;
                            scoring_local_v2_cuda << <gridDimMain, blockDimMain >> > (d_DNA_sequence_r, d_protein_sequence, u_sc_mat, u_ins_mat, u_del_mat, u_t_sc_mat, u_t_ins_mat, u_t_del_mat, N, M, submatrixX * submatrixSide, submatrixY * submatrixSide, submatrixSide);
                            checkCudaErrors(cudaGetLastError());
                        }
                        checkCudaErrors(cudaDeviceSynchronize());
                    }
					timer.Stop();

                    traceV2_1d_check(DNA_sequence_r, protein_sequence, u_sc_mat, u_t_sc_mat, N, M, index_prot, index);
                    cout << "Run DNA: " << index_dna << " Prot: " << index_prot << endl << "Time in ms: " << timer.Elapsed() << endl;
                    top5(index[0], index_prot, index[1], index[2], top_scores, top_i, top_j, top_indeces);
				}
				

                checkCudaErrors(cudaFree(d_DNA_sequence));
                checkCudaErrors(cudaFree(d_protein_sequence));
                checkCudaErrors(cudaFree(d_DNA_sequence_r));

                checkCudaErrors(cudaFree(u_sc_mat));
                checkCudaErrors(cudaFree(u_ins_mat));
                checkCudaErrors(cudaFree(u_del_mat));

                checkCudaErrors(cudaFree(u_t_sc_mat));
                checkCudaErrors(cudaFree(u_t_ins_mat));
                checkCudaErrors(cudaFree(u_t_del_mat));
            }
        }

        if (top == 0) {
            for (int i = 0; i < 5; i++) {
                DNA_sequence = dnaInputs[index_dna];
                protein_sequence = proteinInputs[top_indeces[i]];
                DNA_sequence_r = reverse_complement(DNA_sequence);

                size_t N = DNA_sequence.length();
                size_t M = protein_sequence.length() + 1;

                size_t N_size = N * sizeof(char);
                size_t M_size = M * sizeof(char);
                size_t size = N * M * sizeof(int);
                top_hold = top_scores[0];

                if (mode == 0 && top_hold == top_scores[i]) {
                    cout << proteinIdInputs[top_indeces[i]] << endl << "Sequence: " << top_indeces[i] << endl;
                    int** sc_mat = new int* [N];
                    int** ins_mat = new int* [N];
                    int** del_mat = new int* [N];
                    int** sc_mat_hold = new int* [N];

                    int** t_sc_mat = new int* [N];
                    int** t_ins_mat = new int* [N];
                    int** t_del_mat = new int* [N];
                    int** t_sc_mat_hold = new int* [N];

                    for (int i = 0; i < N; i++) {
                        sc_mat[i] = new int[M]();
                        ins_mat[i] = new int[M]();
                        del_mat[i] = new int[M]();
                        sc_mat_hold[i] = new int[M]();

                        t_sc_mat[i] = new int[M]();
                        t_ins_mat[i] = new int[M]();
                        t_del_mat[i] = new int[M]();
                        t_sc_mat_hold[i] = new int[M]();
                    }

                    init_local_v2(DNA_sequence, protein_sequence, sc_mat, ins_mat, del_mat, t_sc_mat, t_ins_mat, t_del_mat);
                    auto start = steady_clock::now();
                    scoring_local_v2(DNA_sequence, protein_sequence, sc_mat, ins_mat, del_mat, t_sc_mat, t_ins_mat, t_del_mat);
                    auto end = steady_clock::now();
                    auto diff = end - start;
                    traceV2(DNA_sequence, protein_sequence, sc_mat, t_sc_mat, N, M, top_indeces[i], index);
                    cout << endl << "Score: " << top_scores[i] << endl;
                    cout << "Start to End match in Protein: " << myArray[top_indeces[i]][1] << "-" << myArray[top_indeces[i]][2] << endl << endl;
                    cout << "Time in ms: " << duration<double, milli>(diff).count() << endl << endl;

                    for (int i = 0; i < N_size; i++) {
                        delete[] sc_mat[i];
                        delete[] ins_mat[i];
                        delete[] del_mat[i];
                        delete[] t_sc_mat[i];
                        delete[] t_ins_mat[i];
                        delete[] t_del_mat[i];
                        delete[] sc_mat_hold[i];
                        delete[] t_sc_mat_hold[i];
                    }

                    delete[] sc_mat;
                    delete[] ins_mat;
                    delete[] del_mat;
                    delete[] t_sc_mat;
                    delete[] t_ins_mat;
                    delete[] t_del_mat;
                    delete[] sc_mat_hold;
                    delete[] t_sc_mat_hold;

                }
                else if (mode == 1 && top_hold == top_scores[i]) {
                    cout << proteinIdInputs[top_indeces[i]] << endl << "Sequence: " << top_indeces[i] << endl;
                    char* d_DNA_sequence;
                    char* d_protein_sequence;
                    char* d_DNA_sequence_r;

                    int* u_sc_mat;
                    int* u_ins_mat;
                    int* u_del_mat;

                    int* u_t_sc_mat;
                    int* u_t_ins_mat;
                    int* u_t_del_mat;

                    checkCudaErrors(cudaMalloc(&d_DNA_sequence, N_size));
                    checkCudaErrors(cudaMalloc(&d_protein_sequence, M_size));
                    checkCudaErrors(cudaMalloc(&d_DNA_sequence_r, N_size));

                    checkCudaErrors(cudaMemcpy(d_DNA_sequence, DNA_sequence.c_str(), N_size, cudaMemcpyHostToDevice));
                    checkCudaErrors(cudaMemcpy(d_protein_sequence, protein_sequence.c_str(), M_size, cudaMemcpyHostToDevice));
                    checkCudaErrors(cudaMemcpy(d_DNA_sequence_r, DNA_sequence_r.c_str(), N_size, cudaMemcpyHostToDevice));

                    checkCudaErrors(cudaMallocManaged(&u_sc_mat, size));
                    checkCudaErrors(cudaMallocManaged(&u_ins_mat, size));
                    checkCudaErrors(cudaMallocManaged(&u_del_mat, size));

                    checkCudaErrors(cudaMallocManaged(&u_t_sc_mat, size));
                    checkCudaErrors(cudaMallocManaged(&u_t_ins_mat, size));
                    checkCudaErrors(cudaMallocManaged(&u_t_del_mat, size));

                    checkCudaErrors(cudaMemset(u_sc_mat, 0, size));
                    checkCudaErrors(cudaMemset(u_ins_mat, 0, size));
                    checkCudaErrors(cudaMemset(u_del_mat, 0, size));

                    checkCudaErrors(cudaMemset(u_t_sc_mat, 0, size));
                    checkCudaErrors(cudaMemset(u_t_ins_mat, 0, size));
                    checkCudaErrors(cudaMemset(u_t_del_mat, 0, size));

                    dim3 blockDimMain(32, 32);
                    dim3 gridDimMain(1);
                    dim3 blockDimLastRow(1024);
                    dim3 gridDimLastRow(((unsigned int)(M - 1) + blockDimLastRow.x - 1) / blockDimLastRow.x);


                    unsigned int submatrixSide = blockDimMain.x;
                    unsigned int numSubmatrixRows = ((unsigned int)N + submatrixSide - 1) / submatrixSide;
                    unsigned int numSubmatrixCols = ((unsigned int)M + submatrixSide - 1) / submatrixSide;

                    init_local_v2_cuda(DNA_sequence, protein_sequence, u_sc_mat, u_ins_mat, u_del_mat, u_t_sc_mat, u_t_ins_mat, u_t_del_mat, N, M);

                    timer.Start();
                    for (unsigned int diag = 0; diag < numSubmatrixRows + numSubmatrixCols - 1; ++diag) {
                        for (unsigned int submatrixY = std::max(0, (int)diag - (int)(numSubmatrixCols - 1)); submatrixY <= diag && submatrixY < numSubmatrixRows; ++submatrixY) {
                            int submatrixX = diag - submatrixY;
                            scoring_local_v2_cuda << <gridDimMain, blockDimMain >> > (d_DNA_sequence, d_protein_sequence, u_sc_mat, u_ins_mat, u_del_mat, u_t_sc_mat, u_t_ins_mat, u_t_del_mat, N, M, submatrixX * submatrixSide, submatrixY * submatrixSide, submatrixSide);
                            checkCudaErrors(cudaGetLastError());
                        }
                        checkCudaErrors(cudaDeviceSynchronize());
                    }
                    timer.Stop();
                    traceV2_1d(DNA_sequence, protein_sequence, u_sc_mat, u_t_sc_mat, N, M, top_indeces[i], index);
                    cout << endl << "Score: " << top_scores[i] << endl;
                    cout << "Start to End match in Protein: " << myArray[top_indeces[i]][1] << "-" << myArray[top_indeces[i]][2] << endl << endl;
                    cout << "Time in ms: " << timer.Elapsed() << endl << endl;

                    checkCudaErrors(cudaFree(d_DNA_sequence));
                    checkCudaErrors(cudaFree(d_protein_sequence));
                    checkCudaErrors(cudaFree(d_DNA_sequence_r));

                    checkCudaErrors(cudaFree(u_sc_mat));
                    checkCudaErrors(cudaFree(u_ins_mat));
                    checkCudaErrors(cudaFree(u_del_mat));

                    checkCudaErrors(cudaFree(u_t_sc_mat));
                    checkCudaErrors(cudaFree(u_t_ins_mat));
                    checkCudaErrors(cudaFree(u_t_del_mat));

                }
            }
        }
        else if (top == 1) {
            for (int i = 0; i < 5; i++) {
                cout << proteinIdInputs[top_indeces[i]] << endl << "Sequence: " << top_indeces[i] << endl;
                DNA_sequence = dnaInputs[index_dna];
                protein_sequence = proteinInputs[top_indeces[i]];
                DNA_sequence_r = reverse_complement(DNA_sequence);

                size_t N = DNA_sequence.length();
                size_t M = protein_sequence.length() + 1;

                size_t N_size = N * sizeof(char);
                size_t M_size = M * sizeof(char);
                size_t size = N * M * sizeof(int);

                if (mode == 0) {

                    int** sc_mat = new int* [N];
                    int** ins_mat = new int* [N];
                    int** del_mat = new int* [N];
                    int** sc_mat_hold = new int* [N];

                    int** t_sc_mat = new int* [N];
                    int** t_ins_mat = new int* [N];
                    int** t_del_mat = new int* [N];
                    int** t_sc_mat_hold = new int* [N];

                    for (int i = 0; i < N; i++) {
                        sc_mat[i] = new int[M]();
                        ins_mat[i] = new int[M]();
                        del_mat[i] = new int[M]();
                        sc_mat_hold[i] = new int[M]();

                        t_sc_mat[i] = new int[M]();
                        t_ins_mat[i] = new int[M]();
                        t_del_mat[i] = new int[M]();
                        t_sc_mat_hold[i] = new int[M]();
                    }

                    init_local_v2(DNA_sequence, protein_sequence, sc_mat, ins_mat, del_mat, t_sc_mat, t_ins_mat, t_del_mat);
                    auto start = steady_clock::now();
                    scoring_local_v2(DNA_sequence, protein_sequence, sc_mat, ins_mat, del_mat, t_sc_mat, t_ins_mat, t_del_mat);
                    auto end = steady_clock::now();
                    auto diff = end - start;
                    traceV2(DNA_sequence, protein_sequence, sc_mat, t_sc_mat, N, M, top_indeces[i], index);
                    cout << endl << "Score: " << top_scores[i] << endl;
                    cout << "Start to End match in Protein: " << myArray[top_indeces[i]][1] << "-" << myArray[top_indeces[i]][2] << endl << endl;
                    cout << "Time in ms: " << duration<double, milli>(diff).count() << endl << endl;

                    for (int i = 0; i < N_size; i++) {
                        delete[] sc_mat[i];
                        delete[] ins_mat[i];
                        delete[] del_mat[i];
                        delete[] t_sc_mat[i];
                        delete[] t_ins_mat[i];
                        delete[] t_del_mat[i];
                        delete[] sc_mat_hold[i];
                        delete[] t_sc_mat_hold[i];
                    }

                    delete[] sc_mat;
                    delete[] ins_mat;
                    delete[] del_mat;
                    delete[] t_sc_mat;
                    delete[] t_ins_mat;
                    delete[] t_del_mat;
                    delete[] sc_mat_hold;
                    delete[] t_sc_mat_hold;

                }
                else if (mode == 1) {

                    char* d_DNA_sequence;
                    char* d_protein_sequence;
                    char* d_DNA_sequence_r;

                    int* u_sc_mat;
                    int* u_ins_mat;
                    int* u_del_mat;

                    int* u_t_sc_mat;
                    int* u_t_ins_mat;
                    int* u_t_del_mat;

                    checkCudaErrors(cudaMalloc(&d_DNA_sequence, N_size));
                    checkCudaErrors(cudaMalloc(&d_protein_sequence, M_size));
                    checkCudaErrors(cudaMalloc(&d_DNA_sequence_r, N_size));

                    checkCudaErrors(cudaMemcpy(d_DNA_sequence, DNA_sequence.c_str(), N_size, cudaMemcpyHostToDevice));
                    checkCudaErrors(cudaMemcpy(d_protein_sequence, protein_sequence.c_str(), M_size, cudaMemcpyHostToDevice));
                    checkCudaErrors(cudaMemcpy(d_DNA_sequence_r, DNA_sequence_r.c_str(), N_size, cudaMemcpyHostToDevice));

                    checkCudaErrors(cudaMallocManaged(&u_sc_mat, size));
                    checkCudaErrors(cudaMallocManaged(&u_ins_mat, size));
                    checkCudaErrors(cudaMallocManaged(&u_del_mat, size));

                    checkCudaErrors(cudaMallocManaged(&u_t_sc_mat, size));
                    checkCudaErrors(cudaMallocManaged(&u_t_ins_mat, size));
                    checkCudaErrors(cudaMallocManaged(&u_t_del_mat, size));

                    checkCudaErrors(cudaMemset(u_sc_mat, 0, size));
                    checkCudaErrors(cudaMemset(u_ins_mat, 0, size));
                    checkCudaErrors(cudaMemset(u_del_mat, 0, size));

                    checkCudaErrors(cudaMemset(u_t_sc_mat, 0, size));
                    checkCudaErrors(cudaMemset(u_t_ins_mat, 0, size));
                    checkCudaErrors(cudaMemset(u_t_del_mat, 0, size));

                    dim3 blockDimMain(32, 32);
                    dim3 gridDimMain(1);
                    dim3 blockDimLastRow(1024);
                    dim3 gridDimLastRow(((unsigned int)(M - 1) + blockDimLastRow.x - 1) / blockDimLastRow.x);


                    unsigned int submatrixSide = blockDimMain.x;
                    unsigned int numSubmatrixRows = ((unsigned int)N + submatrixSide - 1) / submatrixSide;
                    unsigned int numSubmatrixCols = ((unsigned int)M + submatrixSide - 1) / submatrixSide;

                    init_local_v2_cuda(DNA_sequence, protein_sequence, u_sc_mat, u_ins_mat, u_del_mat, u_t_sc_mat, u_t_ins_mat, u_t_del_mat, N, M);

                    timer.Start();
                    for (unsigned int diag = 0; diag < numSubmatrixRows + numSubmatrixCols - 1; ++diag) {
                        for (unsigned int submatrixY = std::max(0, (int)diag - (int)(numSubmatrixCols - 1)); submatrixY <= diag && submatrixY < numSubmatrixRows; ++submatrixY) {
                            int submatrixX = diag - submatrixY;
                            scoring_local_v2_cuda << <gridDimMain, blockDimMain >> > (d_DNA_sequence, d_protein_sequence, u_sc_mat, u_ins_mat, u_del_mat, u_t_sc_mat, u_t_ins_mat, u_t_del_mat, N, M, submatrixX * submatrixSide, submatrixY * submatrixSide, submatrixSide);
                            checkCudaErrors(cudaGetLastError());
                        }
                        checkCudaErrors(cudaDeviceSynchronize());
                    }
                    timer.Stop();

                    traceV2_1d(DNA_sequence, protein_sequence, u_sc_mat, u_t_sc_mat, N, M, top_indeces[i], index);
                    cout << endl << "Score: " << top_scores[i] << endl;
                    cout << "Start to End match in Protein: " << myArray[top_indeces[i]][1] << "-" << myArray[top_indeces[i]][2] << endl << endl;
                    cout << "Time in ms: " << timer.Elapsed() << endl << endl;

                    checkCudaErrors(cudaFree(d_DNA_sequence));
                    checkCudaErrors(cudaFree(d_protein_sequence));
                    checkCudaErrors(cudaFree(d_DNA_sequence_r));

                    checkCudaErrors(cudaFree(u_sc_mat));
                    checkCudaErrors(cudaFree(u_ins_mat));
                    checkCudaErrors(cudaFree(u_del_mat));

                    checkCudaErrors(cudaFree(u_t_sc_mat));
                    checkCudaErrors(cudaFree(u_t_ins_mat));
                    checkCudaErrors(cudaFree(u_t_del_mat));

                }
            }

        }
    }

    checkCudaErrors(cudaDeviceReset());

    return 0;
}