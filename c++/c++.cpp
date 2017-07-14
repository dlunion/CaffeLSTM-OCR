
//这是一个lstm+cnn的ocr例子
//2017年7月14日 12:23:54
//wish

#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include "classification-c.h"
using namespace std;

#pragma comment(lib, "classification_dll.lib")

vector<char> readFile(const char* file){
	vector<char> data;
	FILE* f = fopen(file, "rb");
	if (!f) return data;

	int len = 0;
	fseek(f, 0, SEEK_END);
	len = ftell(f);
	fseek(f, 0, SEEK_SET);

	if (len > 0){
		data.resize(len);
		fread(&data[0], 1, len, f);
	}
	fclose(f);
	return data;
}

vector<string> loadCodeMap(const char* file){
	ifstream infile(file);
	string line;
	vector<string> out;
	while (std::getline(infile, line)){
		out.push_back(line);
	}
	return out;
}

string getLabel(const vector<string>& labelMap, int index){
	if (index < 0 || index >= labelMap.size())
		return "*";

	return labelMap[index];
}

void main(){
	//禁止caffe输出信息
	disableErrorOutput();

	//注意目录是相对工程上级目录的
	Classifier* classif = createClassifier("deploy.prototxt", "_iter_122659.caffemodel");
	const char* imageFile = "yzm1/5BSRM_9299.png";
	
	vector<string> labelMap = loadCodeMap("label-map.txt");
	vector<char> data = readFile(imageFile);
	if (data.empty()){
		printf("文件不存在么？ %s\n", imageFile);

		releaseClassifier(classif);
		return;
	}

	forward(classif, &data[0], data.size());


	BlobData* premuted_fc = getBlobData(classif, "premuted_fc");
	float* ptr = premuted_fc->list;
	int blank_label = 32;
	int time_step = 19;
	int alphabet_size = 33;
	int prev_label = blank_label;
	string result, result_raw;

	for (int i = 0; i < time_step; ++i){
		float* lin = ptr + i * alphabet_size;
		int predict_label = std::max_element(lin, lin + alphabet_size) - lin;
		float value = lin[predict_label];

		if (predict_label != blank_label && predict_label != prev_label){
			result = result + getLabel(labelMap, predict_label);
		}
		
		result_raw = result_raw + getLabel(labelMap, predict_label);
		prev_label = predict_label;
	}

	releaseBlobData(premuted_fc);
	printf("识别的结果是：\n%s\n%s\n", result.c_str(), result_raw.c_str());
	releaseClassifier(classif);
}