#pragma once

#include "support-common.h"

#ifdef __cplusplus
#include <opencv/cv.h>
#include <vector>
#endif
#include "classification-c.h"

#ifdef __cplusplus
class Caffe_API Classifier {
public:
	Classifier(const char* prototxt_file,
		const char* caffemodel_file,
		float scale_raw = 1,
		const char* mean_file = "",
		int num_means = 0,
		float* means = 0,
		int gpu_id = -1,
		int cach_size = 1);			//cach_size是识别时候的图像个数

	Classifier(const void* prototxt_data,
		int prototxt_data_length,
		const void* caffemodel_data,
		int caffemodel_data_length,
		float scale_raw = 1,
		const char* mean_file = "",
		int num_means = 0,
		float* means = 0,
		int gpu_id = -1,
		int cach_size = 1);

	virtual ~Classifier();

public:
	SoftmaxResult* predictSoftmax(const cv::Mat& img, int top_n = 5);
	MultiSoftmaxResult* predictSoftmax(const std::vector<cv::Mat>& imgs, int top_n = 5);
	BlobData* extfeature(const cv::Mat& img, const char* layer_name = 0);
	void forward(const cv::Mat& img);
	void reshape(int width, int height);

	int input_num(int index = 0);
	int input_channels(int index = 0);
	int input_width(int index = 0);
	int input_height(int index = 0);

	BlobData* getOutputBlob(int index);
	int getOutputBlobCount();
	BlobData* getBlobData(const char* blob_name);

private:
	//Blob<float>
	BlobData* getBlobDataByRawBlob(void* blob);
	void SetMean(const char* mean_file);
	void Predict(const std::vector<cv::Mat>& imgs, std::vector<std::vector<float> >& out);
	void WrapInputLayer(std::vector<cv::Mat>& input_channels);
	void Preprocess(const std::vector<cv::Mat>& imgs, std::vector<cv::Mat>& input_channels);
	void initNetByFile(const char* prototxt, const char* caffemodel);
	void initNetByData(const void* prototxt_data, int prototxt_data_length, const void* caffemodel_data, int caffemodel_data_length);
	void init(float scale_raw = 1, const char* mean_file = "", int num_means = 0, float* means = 0, int gpu_id = -1, int cach_size = 1);

public:
	cv::Size input_geometry_;
	int num_channels_;
	int num_means_;
	float mean_value_[3];
	cv::Mat mean_;
	float scale_raw;
	int cache_size;
	
private:
	void* net_;
};


inline void WPtr<BlobData>::release(BlobData* p){
	releaseBlobData(p);
}

inline void WPtr<SoftmaxResult>::release(SoftmaxResult* p){
	releaseSoftmaxResult(p);
}

inline void WPtr<MultiSoftmaxResult>::release(MultiSoftmaxResult* ptr){
	releaseMultiSoftmaxResult(ptr);
}
#endif