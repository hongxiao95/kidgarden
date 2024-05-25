#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>

#define HXDLL __declspec(dllexport)
#define SWAP(a,b) {int temp=a;a=b;b=temp;}

//编译dll文件的命令是cl /LD xx.c

unsigned int getseeds(){
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (unsigned int)(ts.tv_nsec / 100);
}

//左闭右开
int randbetween(int a, int b){
    int rint = rand();
    return a + (int)(((double)rint / (RAND_MAX + 1.0)) * (b - a));
}

int* shuffle(int* array, int len){
    for(int i = 0; i < len; i++){
        int rindex = randbetween(0, len);
        SWAP(array[i], array[rindex]);
    }
    return array;
}

//初始化range数组，传入时不必要malloc，内部自己分配内存,区间左闭右开
void init_range(int** array, int a, int b){
    int len = b - a;
    *array = (int*)malloc(sizeof(int) * len);
    for(int i = 0; i < len; i++){
        (*array)[i] = a + i;
    }
}

// 用行列索引推导一维展开矩阵的索引
int m_index(int i, int j, int col_size){
    return i * col_size + j;
}

// 计算矩形范围内的和
int sum_rect(int* matrix, int col_size, int y, int x, int h, int w){
    int sum = 0;
    for(int i = y; i < y + h; i++){
        for(int j = x; j < x + w; j++){
            sum += matrix[m_index(i, j, col_size)];
        }
    }
    return sum;
}

// 执行矩阵，将内部全部赋值为0
void execute_rect(int* matrix, int col_size, int y, int x, int h, int w){
    for(int i = y; i < y + h; i++){
        for(int j = x; j < x + w; j++){
            matrix[m_index(i, j, col_size)] = 0;
        }
    }
}


HXDLL int* find_all_seq_matches(int* origin_matrix, int w, int h, int shuffle_index){
    
    srand(getseeds());
    int* matrix = (int*)malloc(sizeof(int) * w * h);
    //最多只可能存在方块数一半的匹配
    int rects_max_count = w * h / 2;
    int** matched_rects = (int**)malloc(sizeof(int*) * rects_max_count);
    for(int i = 0; i < rects_max_count; i++){
        //每个方块是y,x,h,w组成的，需要四个数字,所以分配四个空间
        matched_rects[i] = (int*)malloc(sizeof(int) * 4);
    }

    // 准备打乱的行列索引
    int* row_list = NULL;
    int* col_list = NULL;

    // 应当传入数组变量的地址
    init_range(&row_list, 0 ,h);
    init_range(&col_list, 0, w);

    if(shuffle_index != 0){
        shuffle(row_list, h);
        shuffle(col_list, w);
    }

    for(int _i = 0, i = 0; _i < h; _i++){
        i = row_list[_i];
        int height_max = false;
        for(int _j = 0, j = 0; _j < w; _j++){
            j = col_list[_j];

            if(matrix[m_index(i, j, w)] == 0){
                continue;
            }
            
            bool h_first = randbetween(0,2) < 1;

            //先横向扩展的情况
            if(h_first == true){
                // 确认是否达到最大宽度的flag
                height_max = false;
                // 行列延申时也判断是否以空格子起步
                for(int h_dlt = -i; h_dlt < h - i + 1; h_dlt++){
                    if(matrix[m_index(i, j, w)] == 0){
                        break;
                    }

                    for(int w_dlt = -j; w_dlt < w - j + 1; w_dlt++)
                }
            }



        }
    }

    return NULL;
}

int main(int argc, char** argv){
    int h = 16;
    int w = 10;
    for(int _i = 0, i = 0; _i < h; _i++){
        for(int _j = 0, j = 0; _j < w; _j++){
            printf("(%d,%d,%d,%d)", _i,_j,i,j);
        }
    }
    bool h_first = randbetween(0,2) < 1;
    printf("%d", h_first);


    return 0;
}