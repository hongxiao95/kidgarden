#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define HXDLL __declspec(dllexport)
#define SWAP(a,b) {int SWAPtemp=(a);(a)=(b);(b)=SWAPtemp;}

//����dll�ļ���������cl /LD xx.c

unsigned int getseeds(){
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (unsigned int)(ts.tv_nsec / 100);
}

//����ҿ�
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

//��ʼ��range���飬����ʱ����Ҫmalloc���ڲ��Լ������ڴ�,��������ҿ�
void init_range(int** array, int a, int b){
    int len = b - a;
    *array = (int*)malloc(sizeof(int) * len);
    for(int i = 0; i < len; i++){
        (*array)[i] = a + i;
    }
}

// �����������Ƶ�һάչ�����������
int m_index(int i, int j, int col_size){
    return i * col_size + j;
}

// ������η�Χ�ڵĺ�
int sum_rect(int* matrix, int col_size, int* rect){
    int sum = 0;
    int y = rect[0];
    int x = rect[1];
    int h = rect[2];
    int w = rect[3];
    for(int i = y; i < y + h; i++){
        for(int j = x; j < x + w; j++){
            sum += matrix[m_index(i, j, col_size)];
        }
    }
    return sum;
}

// ִ�о��󣬽��ڲ�ȫ����ֵΪ0
void execute_rect(int* matrix, int col_size, int* rect){
    int y = rect[0];
    int x = rect[1];
    int h = rect[2];
    int w = rect[3];
    for(int i = y; i < y + h; i++){
        for(int j = x; j < x + w; j++){
            matrix[m_index(i, j, col_size)] = 0;
        }
    }
}

/**
 * �������������range������ռ����Ҳ���ڲ����
 * @param side_first <0ʱ����������, > 0 ��������, == 0 �����������
 * @returns ���鳤��
*/
size_t gen_cross_double_range(int** array, int left_stop, int right_stop, int side_first){
    if (side_first == 0){
        side_first = rand() % 2 == 0 ? -1 : 1;
    }
    size_t length = left_stop + right_stop - 2;

    (*array) = (int*)malloc(sizeof(int) * length);


    int insert_index = 0; 
    for(int i = 1; i < max(left_stop, right_stop); i++){
        if(side_first > 0){
            if(i < right_stop){
                (*array)[insert_index++] = i;
            }
            if(i < left_stop){
                (*array)[insert_index++] = -i;
            }
        }else{
            if(i < left_stop){
                (*array)[insert_index++] = -i;
            }
            if(i < right_stop){
                (*array)[insert_index++] = i;
            }
        }
    }
    return length;
}

/**
 * �������Ϊ�����ľ�������
*/
int* fix_rect(y, x, h, w){
    int* rect = (int*)malloc(sizeof(int) * 4);
    rect[0] = y;
    rect[1] = x;
    rect[2] = h;
    rect[3] = w;

    if(h < 0){
        rect[0]-=h;
        rect[2]++;
    }
    if(w < 0){
        rect[1]-=w;
        rect[3]++;
    }

    return rect;
}


HXDLL int** find_all_seq_matches(int* origin_matrix, int rows, int cols, int shuffle_index){
    
    srand(getseeds());
    int* matrix = (int*)malloc(sizeof(int) * cols * rows);
    //���ֻ���ܴ��ڷ�����һ���ƥ��
    int rects_max_count = cols * rows / 2 + 1;
    int** matched_rects = (int**)malloc(sizeof(int*) * rects_max_count);
    int picked_rect_count = 0;

    // ׼�����ҵ���������
    int* row_list = NULL;
    int* col_list = NULL;

    // Ӧ��������������ĵ�ַ
    init_range(&row_list, 0 ,rows);
    init_range(&col_list, 0, cols);

    if(shuffle_index != 0){
        shuffle(row_list, rows);
        shuffle(col_list, cols);
    }

    for(int _i = 0; _i < rows; _i++){
        int i = row_list[_i];
        int height_max = false;
        int width_max = false;
        for(int _j = 0; _j < cols; _j++){
            int j = col_list[_j];

            if(matrix[m_index(i, j, cols)] == 0){
                continue;
            }
            
            bool h_first = rand() % 2 == 1;

            //�Ⱥ�����չ�����
            if(h_first == true){
                // ȷ���Ƿ�ﵽ����ȵ�flag
                height_max = false;
                
                //���������ĸ߶ȷ�Χ
                int* h_range = NULL;
                int h_range_len = gen_cross_double_range(&h_range, i + 1, rows - i + 1,0);

                for(int h_dlt_index = 0; h_dlt_index < h_range_len; h_dlt_index++){
                    int h = h_range[h_dlt_index];
                    // ��������ʱҲ�ж��Ƿ��Կո�����
                    if(matrix[m_index(i, j, cols)] == 0){
                        break;
                    }

                    // ���������Ŀ�ȷ�Χ
                    int* w_range = NULL;
                    int w_range_len = gen_cross_double_range(&w_range, j + 1, cols - j + 1, 0);
                    for(int w_dlt_index = 0; w_dlt_index < w_range_len; w_dlt_index++){
                        int w = w_range[w_dlt_index];

                        if(h == 1 && w == 1){
                            continue;
                        }

                        printf("pre hw fix\n");
                        int* rect = fix_rect(i, j, h, w);
                        int now_sum = sum_rect(matrix, cols, rect);
                        printf("hw sum\n");

                        if(now_sum == 10){
                            matched_rects[picked_rect_count++] = rect;
                            printf("pre hw exec\n");
                            execute_rect(matrix, cols, rect);
                            printf("hw exec\n");
                            height_max = true;
                            break;
                        }else{
                            printf("pre hw free rect\n");
                            free(rect);
                            printf("hw free rect\n");
                        }

                        if(now_sum >= 10 && (w_dlt_index == w_range_len - 1 || abs(w) < abs(w_range[w_dlt_index + 1]))){
                            if(w == 1 && (h_dlt_index == h_range_len - 1 || abs(h) < abs(h_range[h_dlt_index + 1]))){
                                height_max = true;
                            }
                            break;
                        }
                    }
                    free(w_range);
                    if(height_max){
                        break;
                    }
                    
                }
                free(h_range);
            }
            else{
                // ȷ���Ƿ�ﵽ����ȵ�flag
                width_max = false;
                
                //���������Ŀ�ȷ�Χ
                int* w_range = NULL;
                int w_range_len = gen_cross_double_range(&w_range, j + 1, cols - j + 1,0);

                for(int w_dlt_index = 0; w_dlt_index < w_range_len; w_dlt_index++){
                    int w = w_range[w_dlt_index];
                    // ��������ʱҲ�ж��Ƿ��Կո�����
                    if(matrix[m_index(i, j, cols)] == 0){
                        break;
                    }

                    // ���������ĸ߶ȷ�Χ
                    int* h_range = NULL;
                    int h_range_len = gen_cross_double_range(&h_range, i + 1, cols - i + 1, 0);
                    for(int h_dlt_index = 0; h_dlt_index < h_range_len; h_dlt_index++){
                        int h = h_range[h_dlt_index];

                        if(h == 1 && w == 1){
                            continue;
                        }

                        printf("pre wh fix\n");
                        int* rect = fix_rect(i, j, h, w);
                        int now_sum = sum_rect(matrix, cols, rect);
                        printf("wh sum\n");

                        if(now_sum == 10){
                            matched_rects[picked_rect_count++] = rect;
                            printf("pre wh exec\n");
                            execute_rect(matrix, cols, rect);
                            printf("wh exec\n");
                            height_max = true;
                            break;
                        }else{
                            printf("pre wh free rect\n");
                            free(rect);
                            printf("wh free rect\n");
                        }

                        if(now_sum >= 10 && (h_dlt_index == h_range_len - 1 || abs(h) < abs(h_range[h_dlt_index + 1]))){
                            if(h == 1 && (w_dlt_index == w_range_len - 1 || abs(w) < abs(w_range[w_dlt_index + 1]))){
                                width_max = true;
                            }
                            break;
                        }
                    }
                    free(h_range);
                    if(width_max){
                        break;
                    }
                }
                free(w_range);
            }
        }
    }
    free(matrix);
    free(row_list);
    free(col_list);
    int* ends_symbol = (int*)malloc(sizeof(int) * 4);
    ends_symbol[0] = -1;
    ends_symbol[1] = -1;
    ends_symbol[2] = -1;
    ends_symbol[3] = -1;

    matched_rects[picked_rect_count] = ends_symbol;

    return matched_rects;
}

int main(int argc, char** argv){
    int* array = (int*)malloc(sizeof(int) * 160);

    printf("len: %zd", sizeof(array));

    free(array);

    return 0;
}