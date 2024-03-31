
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include <stdio.h>
#include "math_functions.h"
#include <cmath>
#include <stdio.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

struct ray
{
	float x;
	float y;
	float z;
	//NORMALIZADOOOOOO!
	float d_x;
	float d_y;
	float d_z;
};

struct light
{
	float x;
	float y;
	float z;
	float radio;
	float intensity;
	float r;
	float g;
	float b;
};

struct esfera
{
	float x;
	float y;
	float z;
	float radio;
	//material
	char r;
	char g;
	char b;
	float difrac;
	float refrac;
};

struct vector3
{
	float x;
	float y;
	float z;
};
__device__ void normalize(vector3* vector) {
	float aux = sqrtf(vector->x * vector->x + vector->y * vector->y + vector->z * vector->z);

	vector->x /= aux;
	vector->y /= aux;
	vector->z /= aux;
}

__device__ vector3 phongShading(light* luz, vector3* point, vector3* normal, vector3* dir_camera, vector3* color) {
	float ambiental = 0.6f;
	float difuso = 0.5f;
	float especular = 0.3f;
	float brillantez = 20;

	vector3 colorSalida;
	colorSalida.x = 0;
	colorSalida.y = 0;
	colorSalida.z = 0;
	// Calcular ambiental

	colorSalida.x += color->x * ambiental;
	colorSalida.y += color->y * ambiental;
	colorSalida.z += color->z * ambiental;
	// difisuiasion
	for (int index = 0; index < 2; index++) {
		vector3 luz_vect;
		luz_vect.x = luz[index].x - point->x;
		luz_vect.y = luz[index].y - point->y;
		luz_vect.z = luz[index].z - point->z;
		normalize(&luz_vect);

		float dot_difuso = luz_vect.x * normal->x + luz_vect.y * normal->y + luz_vect.z * normal->z;
		if (dot_difuso > 0) {
			colorSalida.x += dot_difuso * color->x * difuso;
			colorSalida.y += dot_difuso * color->y * difuso;
			colorSalida.z += dot_difuso * color->z * difuso;

			// Especular
			vector3 rVect;
			rVect.x = luz_vect.x - 2.0f * (dot_difuso)*normal->x;
			rVect.y = luz_vect.y - 2.0f * (dot_difuso)*normal->y;
			rVect.z = luz_vect.z - 2.0f * (dot_difuso)*normal->z;
			// dot(v, r)
			float dotVR = rVect.x * dir_camera->x + rVect.y * dir_camera->y + rVect.z * dir_camera->z;

			dotVR = powf(dotVR, brillantez);

			colorSalida.x += dotVR * color->x * especular;
			colorSalida.y += dotVR * color->y * especular;
			colorSalida.z += dotVR * color->z * especular;
		}
	}
	
	colorSalida.x = min(255, (int)roundf(colorSalida.x));
	colorSalida.y = min(255, (int)roundf(colorSalida.y));
	colorSalida.z = min(255, (int)roundf(colorSalida.z));
	return colorSalida;
}
//__device__ __host__ bool secondaryRays() {
//
//}
__device__ __host__ bool sphereIntersection(esfera* esf, ray* rayo, float* dist)
{
	float vX, vY, vZ;
	float discriminante;

	float a, b, c;

	vX = rayo->x - esf->x;
	vY = rayo->y - esf->y;
	vZ = rayo->z - esf->z;

	a = (rayo->d_x * rayo->d_x + rayo->d_y * rayo->d_y + rayo->d_z * rayo->d_z);
	b = 2.0f * (vX * rayo->d_x + vY * rayo->d_y + vZ * rayo->d_z);
	c = (vX * vX + vY * vY + vZ * vZ) - (esf->radio * esf->radio);
	discriminante = (b * b) - (4 * a * c);
	if (discriminante < 0.0f)
		return false;
	else {

		*dist = (-b - sqrtf(discriminante)) / (2.0f * a);
		return true;
	}
	return false;
}



__global__ void rayCasting(int width, int height, vector3* esquina, esfera* esferas, light* luz, vector3* camera, uchar* output, float inc_x, float inc_y)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	if (i < width && j < height)
	{
		int idx = i * width * 3 + j * 3;
		ray primary;
		vector3 dest;
		dest.x = 1;
		dest.y = esquina->y + inc_y * j;
		dest.z = esquina->z + inc_x * i;

		primary.x = camera->x;
		primary.y = camera->y;
		primary.z = camera->z;

		primary.d_x = dest.x - primary.x;
		primary.d_y = dest.y - primary.y;
		primary.d_z = dest.z - primary.z;

		float aux = sqrtf(primary.d_x * primary.d_x + primary.d_y * primary.d_y + primary.d_z * primary.d_z);

		primary.d_x /= aux;
		primary.d_y /= aux;
		primary.d_z /= aux;
		float dist = 0;
		if (sphereIntersection(esferas, &primary, &dist))
		{
			vector3 interPoint;
			interPoint.x = primary.d_x * dist + primary.x;
			interPoint.y = primary.d_y * dist + primary.y;
			interPoint.z = primary.d_z * dist + primary.z;

			vector3 normal;
			normal.x = interPoint.x - esferas->x;
			normal.y = interPoint.y - esferas->y;
			normal.z = interPoint.z - esferas->z;
			normalize(&normal);

			vector3 colorInicio;
			colorInicio.x = esferas->r;
			colorInicio.y = esferas->g;
			colorInicio.z = esferas->b;
			vector3 cameraVect;
			cameraVect.x = camera->x - interPoint.x;
			cameraVect.y = camera->y - interPoint.y;
			cameraVect.z = camera->z - interPoint.z;
			normalize(&cameraVect);

			colorInicio = phongShading(luz, &interPoint, &normal, &cameraVect, &colorInicio);
			output[idx] = colorInicio.y;
			output[idx + 1] = colorInicio.x;
			output[idx + 2] = colorInicio.z;
		}
		else
		{
			output[idx] = 30;
			output[idx + 1] = 30;
			output[idx + 2] = 30;
		}
		//output[idx] = 255;


	}

}

__global__ void multipleRayCasting(int width, int height, vector3* esquina, esfera * esferas, light* luz, vector3* camera, uchar* output, float inc_x, float inc_y)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	if (i < width && j < height)
	{
		int idx = i * width * 3 + j * 3;
		ray primary;
		vector3 dest;
		dest.x = 1;
		dest.y = esquina->y + inc_y * j;
		dest.z = esquina->z + inc_x * i;

		primary.x = camera->x;
		primary.y = camera->y;
		primary.z = camera->z;

		primary.d_x = dest.x - primary.x;
		primary.d_y = dest.y - primary.y;
		primary.d_z = dest.z - primary.z;

		float aux = sqrtf(primary.d_x * primary.d_x + primary.d_y * primary.d_y + primary.d_z * primary.d_z);

		primary.d_x /= aux;
		primary.d_y /= aux;
		primary.d_z /= aux;
		float dist = 0;
		vector3 colorInicio[3];
		for (int index = 0; index < 3; index++) {
			if (sphereIntersection(&esferas[index], &primary, &dist))
			{
				vector3 interPoint;
				interPoint.x = primary.d_x * dist + primary.x;
				interPoint.y = primary.d_y * dist + primary.y;
				interPoint.z = primary.d_z * dist + primary.z;

				vector3 normal;
				normal.x = interPoint.x - esferas[index].x;
				normal.y = interPoint.y - esferas[index].y;
				normal.z = interPoint.z - esferas[index].z;
				normalize(&normal);
				colorInicio[index].x = esferas[index].r;
				colorInicio[index].y = esferas[index].g;
				colorInicio[index].z = esferas[index].b;
				vector3 cameraVect;
				cameraVect.x = camera->x - interPoint.x;
				cameraVect.y = camera->y - interPoint.y;
				cameraVect.z = camera->z - interPoint.z;
				normalize(&cameraVect);

				colorInicio[index] = phongShading(luz, &interPoint, &normal, &cameraVect, &colorInicio[index]);
				output[idx] = colorInicio[index].y;
				output[idx + 1] = colorInicio[index].x;
				output[idx + 2] = colorInicio[index].z;
			}
			else if (!sphereIntersection(&esferas[0], &primary, &dist) && !sphereIntersection(&esferas[1], &primary, &dist) && !sphereIntersection(&esferas[2], &primary, &dist))
			{
				output[idx] = 30;
				output[idx + 1] = 30;
				output[idx + 2] = 30;
			}
		}
		
	}

}


int main()
{
	//creamos camara
	vector3 camera;
	camera.x = 0;
	camera.y = 0;
	camera.z = 0;

	vector3 cent_img;
	cent_img.x = 1;
	cent_img.y = 0;
	cent_img.z = 0;

	int width = 800, height = 800;
	float tam_imgX = 2, tam_imgY = 2;

	esfera esf1;
	esf1.x = 10;
	esf1.y = 0;
	esf1.z = 1;
	esf1.r = 168;
	esf1.g = 50;
	esf1.b = 123;
	esf1.radio = 2;

	esfera esf2;
	esf2.x = 10;
	esf2.y = -5;
	esf2.z = -3;
	esf2.r = 154;
	esf2.g = 74;
	esf2.b = 154;
	esf2.radio = 2;

	esfera esf3;
	esf3.x = 10;
	esf3.y = 1;
	esf3.z = -4;
	esf3.r = 51;
	esf3.g = 255;
	esf3.b = 0;
	esf3.radio = 2;
	esfera esferaArray[3] = {esf1, esf2, esf3};

	light luz1;
	luz1.x = 10;
	luz1.y = 4;
	luz1.z = 1;
	luz1.radio = 3;

	light luz2;
	luz2.x = 10;
	luz2.y = 2;
	luz2.z = 1;
	luz2.radio = 2;

	light lightArray[2] = {luz1, luz2};

	vector3 esquina_img;
	esquina_img.x = cent_img.x;
	esquina_img.y = tam_imgY / 2.0f;
	esquina_img.z = -tam_imgX / 2.0f;



	float inc_x = tam_imgX / width;
	float inc_y = tam_imgY / height;

	//agregamos desf al cent
	esquina_img.y -= inc_y / 2.0f;
	esquina_img.z += inc_x / 2.0f;


	vector3* camera_dev;
	esfera* esferas_dev;
	vector3* esquina_dev;
	uchar* img_dev;
	light* luz_dev;
	esfera* arrayd_dev;
	light* light_array_dev;

	dim3 threads(16, 16);
	dim3 blocks(ceil((float)width / (float)threads.x), ceil((float)height / (float)threads.y));

	cudaMalloc(&camera_dev, sizeof(vector3));
	cudaMalloc(&img_dev, width * height * 3);
	cudaMalloc(&esferas_dev, sizeof(esfera));
	cudaMalloc(&esquina_dev, sizeof(vector3));
	cudaMalloc(&luz_dev, sizeof(light));
	cudaMalloc(&light_array_dev, sizeof(light)*2);
	cudaMalloc(&arrayd_dev, sizeof(esfera) * 3);

	cudaMemcpy(camera_dev, &camera, sizeof(vector3), cudaMemcpyHostToDevice);
	cudaMemcpy(esferas_dev, &esf1, sizeof(esfera), cudaMemcpyHostToDevice);
	cudaMemcpy(esquina_dev, &esquina_img, sizeof(vector3), cudaMemcpyHostToDevice);
	cudaMemcpy(luz_dev, &luz1, sizeof(light), cudaMemcpyHostToDevice);
	cudaMemcpy(light_array_dev, lightArray, sizeof(light) * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(arrayd_dev, esferaArray, sizeof(esfera) * 3, cudaMemcpyHostToDevice);

	//rayCasting << <blocks, threads >> > (width, height, esquina_dev, esferas_dev, luz_dev, camera_dev, img_dev, inc_x, -inc_y);
	multipleRayCasting << <blocks, threads >> > (width, height, esquina_dev, arrayd_dev, light_array_dev, camera_dev, img_dev, inc_x, -inc_y);

	cv::Mat frame = cv::Mat(cv::Size(width, height), CV_8UC3);
	cudaMemcpy(frame.ptr(), img_dev, width * height * 3, cudaMemcpyDeviceToHost);

	cv::imshow("salida", frame);
	cv::waitKey(0);

}