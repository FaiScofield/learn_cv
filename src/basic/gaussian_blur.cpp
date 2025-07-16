#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <iostream>
#include <string>
#include <chrono>

#if WIN32
#include <opencv2/highgui.hpp>
#include "NEON_2_SSE.h"
#else
#include <unistd.h>	// usleep
#endif

using namespace cv;
using namespace std;

#define ENABLE_TEST_PROC_OPENCV 			1
#define ENABLE_TEST_PROC_SELF_DEFINE 		1

#define ENABLE_TEST_CV_PARALLEL_FOR 		1
#define ENABLE_TEST_SIMD_OPEN				1

#define ENABLE_TEST_SLEEP_BETWEEN_PROC		0
#define ENABLE_TEST_WIRTE_OUT_RESULT		(1 && WIN32)

char g_filename[256] = {0};
int g_multi_thread_num = 1;

////---- ReadData
int ReadData(const char* fileName, int nSize, unsigned char* pData)
{
	FILE* fp = NULL;
	fp = fopen(fileName, "r+b");
	if (fp == NULL)
	{
		printf("Read %s failure !\n", fileName);
		return -1;
	}
	fread(pData, sizeof(unsigned char), nSize, fp);
	fclose(fp);
	return  0;
}

void GaussianBlur3x3(uchar* pSrc, int nImgHgt, int nImgWid, int nSrcStride, uchar* pDst, int nDstStride)
{
	// type define
	typedef uchar  SrcType;		// src image type
	typedef uchar  DstType;		// dst image type
	typedef ushort OprType;		// tmp operand data type

#if 0
	for (int y = 1; y < nImgHgt - 1; y++)
	{
		SrcType* pSrcRow0 = pSrc + (y - 1) * nSrcStride;
		SrcType* pSrcRow1 = pSrc + (y + 0) * nSrcStride;
		SrcType* pSrcRow2 = pSrc + (y + 1) * nSrcStride;
		DstType* pDstRow  = pDst + (y + 0) * nDstStride;
		for (int x = 1; x < nImgWid - 1; x++)
		{
			OprType sum0 = pSrcRow0[x-1] + 2 * pSrcRow0[x] + pSrcRow0[x+1];
			OprType sum1 = pSrcRow1[x-1] + 2 * pSrcRow1[x] + pSrcRow1[x+1];
			OprType sum2 = pSrcRow2[x-1] + 2 * pSrcRow2[x] + pSrcRow2[x+1];
			pDstRow[x] = (sum0 + 2 * sum1 + sum2 + 8) >> 4;
		}
	}
	return;
#endif

	//! NOTE: Desired result won't be required due to data coverage when (pSrc == pDst) if using multi-thread !!
#if ENABLE_TEST_CV_PARALLEL_FOR
	cv::parallel_for_(
		cv::Range(0, nImgHgt),	// loop range, the 1st argument
		[&](const cv::Range& range)
		{
			const int nStaRow = range.start;
			const int nEndRow = range.end;
#else
	const int nStaRow = 0;
	const int nEndRow = nImgHgt;
#endif

	const int nWinSize = 3;
	const int nRadius  = nWinSize / 2;
	const int nLength0 = nImgWid - nRadius;

	// kernel[9] = [1, 2, 1; 2, 4, 2; 1, 2, 1], sum to 16, 4 bit fixed-point
	const uchar coef0 = 1;
	const uchar coef1 = 2;
	// const uchar coef2 = 1;

	const int nRingBufStep = 4800 * 1;
	OprType   tmpRingBuf[nRingBufStep * nWinSize];
	OprType*  pRingBufStart = tmpRingBuf;
	OprType*  pRingBufRows[nWinSize] = {NULL};

	int sy = MAX(0, nStaRow - nRadius);
	for (int y = nStaRow; y < nEndRow; ++y)
	{
		const int syLimit = MIN(y + nRadius, nImgHgt - 1);
		for (; sy <= syLimit; sy++)
		{
			const SrcType* pSrcRow  = (SrcType*)pSrc + sy * nSrcStride;
				  OprType* pRingRow = pRingBufStart + ((sy + nRadius) % nWinSize) * nRingBufStep;

			// left
			pRingRow[0] = 2 * coef0 * pSrcRow[1] + coef1 * pSrcRow[0];

			// middle
			int x = 1;
		#if ENABLE_TEST_SIMD_OPEN
			const SrcType* pSrc0 = pSrcRow + 0;
			const SrcType* pSrc1 = pSrcRow + 1;
			const SrcType* pSrc2 = pSrcRow + 2;
				  OprType* pDst  = pRingRow + 1;
			uint8x16_t v_r0, v_r1, v_r2;
			uint16x8_t v_d0, v_d1;
			for (; x <= nLength0 - 16; x += 16)
			{
				v_r0  = vld1q_u8(pSrc0);
				v_r1  = vld1q_u8(pSrc1);
				v_r2  = vld1q_u8(pSrc2);
				v_d0  = vaddl_u8(vget_low_u8(v_r0), vget_low_u8(v_r2));
				v_d1  = vaddl_u8(vget_high_u8(v_r0), vget_high_u8(v_r2));
				v_d0  = vaddq_u16(v_d0, vshll_n_u8(vget_low_u8(v_r1), 1));
				v_d1  = vaddq_u16(v_d1, vshll_n_u8(vget_high_u8(v_r1), 1));
				vst1q_u16(pDst + 0, v_d0);
				vst1q_u16(pDst + 8, v_d1);
				pSrc0 += 16;
				pSrc1 += 16;
				pSrc2 += 16;
				pDst  += 16;
			}
			if (x <= nLength0 - 8)
			{
				uint8x8_t v_r0 = vld1_u8(pSrc0);
				uint8x8_t v_r1 = vld1_u8(pSrc1);
				uint8x8_t v_r2 = vld1_u8(pSrc2);
				vst1q_u16(pDst, vaddq_u16(vaddl_u8(v_r0, v_r2), vshll_n_u8(v_r1, 1)));
				x += 8;
			}
		#endif
			for (; x < nLength0; ++x) {
				pRingRow[x] = coef0 * (pSrcRow[x-1] + pSrcRow[x+1]) + coef1 * pSrcRow[x];
			}

			// right
			pRingRow[x] = 2 * coef0 * pSrcRow[x-1] + coef1 * pSrcRow[x];
		}

		// sort tmpBuf by row index
		for (int k = 0; k < nWinSize; k++) {
			pRingBufRows[k] = pRingBufStart + ((y + k) % nWinSize) * nRingBufStep;
		}
		OprType* pRow0 = pRingBufRows[0];
		OprType* pRow1 = pRingBufRows[1];
		OprType* pRow2 = pRingBufRows[2];
		if (y == 0) {
			pRow0 = pRow2;
		} else if (y == nImgHgt - 1) {
			pRow2 = pRow0;
		}

		DstType* pDstRow = (DstType*)pDst + y * nDstStride;
		int x = 0;
	#if ENABLE_TEST_SIMD_OPEN
		uint16x8_t v_r0, v_r1, v_r2;
		uint8x8_t  v_d0, v_d1;
		for (; x <= nImgWid - 16; x += 16)
		{
			v_r0 = vld1q_u16(pRow0 + x);
			v_r2 = vld1q_u16(pRow2 + x);
			v_r1 = vld1q_u16(pRow1 + x);
			v_r0 = vaddq_u16(v_r0, v_r2);
			v_r1 = vshlq_n_u16(v_r1, 1);
			v_d0  = vrshrn_n_u16(vaddq_u16(v_r0, v_r1), 4);
			v_r0 = vld1q_u16(pRow0 + x + 8);
			v_r2 = vld1q_u16(pRow2 + x + 8);
			v_r1 = vld1q_u16(pRow1 + x + 8);
			v_r0 = vaddq_u16(v_r0, v_r2);
			v_r1 = vshlq_n_u16(v_r1, 1);
			v_d1 = vrshrn_n_u16(vaddq_u16(v_r0, v_r1), 4);
			vst1q_u8(pDstRow + x, vcombine_u8(v_d0, v_d1));
		}
		if (x <= nImgWid - 8)
		{
			v_r0 = vld1q_u16(pRow0 + x);
			v_r2 = vld1q_u16(pRow2 + x);
			v_r1 = vld1q_u16(pRow1 + x);
			v_r0 = vaddq_u16(v_r0, v_r2);
			v_r1 = vshlq_n_u16(v_r1, 1);
			v_d0  = vrshrn_n_u16(vaddq_u16(v_r0, v_r1), 4);
			vst1_u8(pDstRow + x, v_d0);
			x += 8;
		}
	#endif
		for (; x < nImgWid; ++x) {
			pDstRow[x] = (coef0 * (pRow0[x] + pRow2[x]) + coef1 * pRow1[x] + 8) >> 4;
		}
	}

#if ENABLE_TEST_CV_PARALLEL_FOR
		},						// lambda definition, the 2nd argument
		g_multi_thread_num		// thread number, the 3rd argument
	);							// cv::parallel_for_
#endif
}

int main(int argc, char const *argv[])
{
	cout << "[TEST] ========== TEST Filter ==========" << endl;

#if WIN32
	const string imgFile = "G:/expfusion/bbkData/20210208104003/4160_3120_0_NV21.yuv";
#else
	if (argc < 2) {
		cout << "[TEST] Usage: exe_file <image_yuv_file.yuv> [kernel_size] [core_num]" << endl;
		cout << "[TEST] input image type: size(4032x3000), fromat(NV21)" << endl;
		cout << "[TEST] input core_num: 1~8, default(1)" << endl;
		return -1;
	}
	const string imgFile(argv[1]);
	cout << "[TEST] input file: " << imgFile << endl;
#endif

	// set kernel_size
	int kernel_size = 3;
	if (argc >= 3) {
		kernel_size = atoi(argv[2]);
	}
	cout << "[TEST] kernel_size: " << kernel_size << endl;

	// set core_num
	int core_num = 6;
	if (argc >= 4) {
		core_num = atoi(argv[3]);
	}
#if !ENABLE_TEST_CV_PARALLEL_FOR
	core_num = 1;
#endif
	// core_num = CLIP(core_num, 1, 8);
	cout << "[TEST] core_num used: " << core_num << endl;
	setNumThreads(core_num);
	g_multi_thread_num = core_num;

	cout << "[TEST] Use " << getNumThreads() << " threads for cv parallel computation..." << endl;

#if ENABLE_TEST_SIMD_OPEN
	cout << "[TEST] Using SIMD..!! " << endl;
#endif

	/// 
	int cn = 1;
	const int padSz  = 3;
	const int imgHgt = 3120;
	const int imgWid = 4160;
	// const int imgHgt = 3000;
	// const int imgWid = 4032;
	const int dataSize = imgHgt * imgWid; // *3/2  	//! only read Y channel data

	/// read yuv image
	Mat src_gray(imgHgt, imgWid, CV_8UC1);
	ReadData(imgFile.c_str(), dataSize, src_gray.data);
	Mat src = src_gray;

#if WIN32 && 1
	Mat show;
	resize(src, show, Size(1008, 750));
	imshow("Org Img", show);
	waitKey(10);
#endif

#ifdef _DEBUG
	const int nRepeatTime = 1;
	setNumThreads(1);
	cout << "[TEST] Use " << getNumThreads() << " threads for _DEBUG mode!" << endl;
#elif WIN32
	const int nRepeatTime = 5;
#else
	const int nRepeatTime = 30;
#endif

	/// orderFilter compare
	double tatalCost = 0.;
	int tatalCall = 0;

	Mat dst1(src.size(), src.type());
	Mat dst2(src.size(), src.type());
	Mat dst3(src.size(), src.type());

	// by opencv
#if ENABLE_TEST_PROC_OPENCV
	ocl::setUseOpenCL(false);	// avoid calling OpenCL functions

	for (int t = 0; t < nRepeatTime; ++t)
	{
#if !WIN32 && ENABLE_TEST_SLEEP_BETWEEN_PROC
		usleep(0.01 * 1e6);	// sleep 10ms
#endif
		chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
		cv::GaussianBlur(src, dst1, cv::Size(3, 3), 1);
		chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
		double cost = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count() * 1000;
		printf("[TEST] Time cost for %dx%d gaussianBlur: %.2f[ms]\n", kernel_size, kernel_size, cost);

		if (t > 1) {
			tatalCost += cost;
			tatalCall++;
		}
	}
	printf("[TEST] Time cost Mean by CV: %.2f[ms]\n", tatalCost/tatalCall);

#if WIN32 && ENABLE_TEST_WIRTE_OUT_RESULT
	snprintf(g_filename, 256, "G:/expfusion/exp/test_bf_by_cv.png");
	imwrite(g_filename, dst1);
#endif

#endif	// ENABLE_TEST_PROC_OPENCV

#if !WIN32 && ENABLE_TEST_SLEEP_BETWEEN_PROC
	usleep(0.1 * 1e6);
#endif

	// by self-define
#if ENABLE_TEST_PROC_SELF_DEFINE
	dst1.setTo(0);
	dst2.setTo(0);
	dst3.setTo(0);
	tatalCost = tatalCall = 0;
	for (int t = 0; t < nRepeatTime; ++t) {
#if !WIN32 && ENABLE_TEST_SLEEP_BETWEEN_PROC
		usleep(0.01 * 1e6);	// sleep 10ms
#endif
		chrono::steady_clock::time_point t1, t2, t3, t4; 
		switch (kernel_size)
		{
		case 3:
			t1 = chrono::steady_clock::now();
			GaussianBlur3x3(src.data, src.rows, src.cols, src.step1(), dst1.data, dst1.step1());
			t2 = chrono::steady_clock::now();
			break;
		case 5:
		default:
			t1 = chrono::steady_clock::now();
			// GaussianBlur5x5(src.data, src.rows, src.cols, src.step1(), dst1.data, dst1.step1());
			t2 = chrono::steady_clock::now();
			break;
		}
		double cost = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count() * 1000;
		printf("[TEST] Time cost for %dx%d gaussianBlur: %.2f[ms]\n", kernel_size, kernel_size, cost);

		if (t > 1) {
			tatalCost += cost;
			tatalCall++;
		}
	}
	printf("[TEST] Time cost Mean by Self: %.2f[ms]\n", tatalCost/tatalCall);

#if WIN32 && ENABLE_TEST_WIRTE_OUT_RESULT
	snprintf(g_filename, 256, "G:/expfusion/exp/test_bf_by_self.png");
	imwrite(g_filename, dst1);
#endif

#endif	// ENABLE_TEST_PROC_SELF_DEFINE

	cout << "[TEST] Done." << endl;
	return 0;
}