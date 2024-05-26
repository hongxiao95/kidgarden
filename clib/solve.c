#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define HXDLL __declspec(dllexport)
#define SWAP(a,b) {int SWAPtemp=(a);(a)=(b);(b)=SWAPtemp;}

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

// 执行矩阵，将内部全部赋值为0
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
 * 生成正负交替的range，数组空间分配也在内部完成
 * @param side_first <0时，负数优先, > 0 正数优先, == 0 随机决定优先
 * @returns 数组长度
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
 * 修正宽高为负数的矩形区域
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
    for (size_t i = 0; i < rows * cols; i++)
    {
        matrix[i] = origin_matrix[i];
    }
    
    //最多只可能存在方块数一半的匹配
    int rects_max_count = (cols * rows) / 2 + 1;
    printf("rects max : %d\n", rects_max_count);
    int** matched_rects = (int**)malloc(sizeof(int*) * rects_max_count);
    int picked_rect_count = 0;

    // 准备打乱的行列索引
    int* row_list = NULL;
    int* col_list = NULL;

    // 初始化range应当传入数组变量的地址
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

            //先横向扩展的情况
            if(h_first == true){
                // 确认是否达到最大宽度的flag
                height_max = false;
                
                //建立搜索的高度范围
                int* h_range = NULL;
                int h_range_len = gen_cross_double_range(&h_range, i + 1, rows - i + 1,0);
                // printf("\nfor i:%d, j:%d, h_range_len: %d, has h:", i, j, h_range_len);
                // for (size_t hi = 0; hi < h_range_len; hi++)
                // {
                //     printf("%d, ", h_range[hi]);
                // }
                // printf("\n");
                
                for(int h_dlt_index = 0; h_dlt_index < h_range_len; h_dlt_index++){
                    int h = h_range[h_dlt_index];
                    // 行列延申时也判断是否以空格子起步
                    if(matrix[m_index(i, j, cols)] == 0){
                        break;
                    }

                    // 建立搜索的宽度范围
                    int* w_range = NULL;
                    int w_range_len = gen_cross_double_range(&w_range, j + 1, cols - j + 1, 0);
                    // printf("\nfor i:%d, j:%d, w_range_len: %d, has w:", i, j, w_range_len);
                    // for (size_t wi = 0; wi < w_range_len; wi++)
                    // {
                    //     printf("%d, ", w_range[wi]);
                    // }
                    // printf("\n");
                    for(int w_dlt_index = 0; w_dlt_index < w_range_len; w_dlt_index++){
                        int w = w_range[w_dlt_index];

                        if(h == 1 && w == 1){
                            continue;
                        }

                        // printf("pre hw fix\n");
                        printf("before fix: (%d, %d, %d, %d) -> ", i, j, h, w);
                        int* rect = fix_rect(i, j, h, w);
                        printf("after fix: (%d, %d, %d, %d)\n\n", rect[0], rect[1], rect[2], rect[3]);
                        int now_sum = sum_rect(matrix, cols, rect);
                        // printf("hw sum\n");

                        if(now_sum == 10){
                            matched_rects[picked_rect_count++] = rect;
                            // printf("pre hw exec\n");
                            execute_rect(matrix, cols, rect);
                            // printf("hw exec\n");
                            height_max = true;
                            break;
                        }else{
                            // printf("pre hw free rect\n");
                            free(rect);
                            // printf("hw free rect\n");
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
                // 确认是否达到最大宽度的flag
                width_max = false;
                
                //建立搜索的宽度范围
                int* w_range = NULL;
                int w_range_len = gen_cross_double_range(&w_range, j + 1, cols - j + 1,0);

                for(int w_dlt_index = 0; w_dlt_index < w_range_len; w_dlt_index++){
                    int w = w_range[w_dlt_index];
                    // 行列延申时也判断是否以空格子起步
                    if(matrix[m_index(i, j, cols)] == 0){
                        break;
                    }

                    // 建立搜索的高度范围
                    int* h_range = NULL;
                    int h_range_len = gen_cross_double_range(&h_range, i + 1, cols - i + 1, 0);
                    for(int h_dlt_index = 0; h_dlt_index < h_range_len; h_dlt_index++){
                        int h = h_range[h_dlt_index];

                        if(h == 1 && w == 1){
                            continue;
                        }

                        // printf("pre wh fix\n");
                        int* rect = fix_rect(i, j, h, w);
                        int now_sum = sum_rect(matrix, cols, rect);
                        // printf("wh sum\n");

                        if(now_sum == 10){
                            matched_rects[picked_rect_count++] = rect;
                            // printf("pre wh exec\n");
                            execute_rect(matrix, cols, rect);
                            // printf("wh exec\n");
                            height_max = true;
                            break;
                        }else{
                            // printf("pre wh free rect\n");
                            free(rect);
                            // printf("wh free rect\n");
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
    int array[160] = {2, 5, 9, 9, 2, 1, 8, 4, 2, 9, 5, 4, 9, 5, 3, 8, 2, 9, 6, 9, 2, 9, 7, 8, 6, 3, 3, 8, 7, 6, 8, 8, 9, 2, 3, 1, 6, 8, 9, 5, 2, 2, 5, 7, 2, 4, 3, 9, 5, 5, 7, 3, 8, 1, 6, 7, 8, 7, 3, 1, 7, 8, 6, 1, 2, 5, 1, 8, 1, 4, 4, 3, 7, 7, 6, 8, 1, 1, 6, 2, 8, 4, 9, 4, 7, 7, 1, 1, 6, 8, 2, 4, 5, 3, 9, 7, 5, 3, 6, 9, 4, 2, 5, 1, 9, 1, 6, 5, 1, 7, 4, 1, 4, 6, 3, 7, 9, 9, 9, 1, 4, 5, 9, 3, 1, 2, 7, 4, 3, 5, 3, 9, 1, 6, 2, 2, 4, 3, 8, 3, 4, 4, 3, 4, 4, 1, 9, 4, 2, 6, 4, 3, 5, 1, 7, 6, 8, 1, 5, 9};
    
    int** rects = NULL;

    rects = find_all_seq_matches(array, 16, 10, 1);
    printf("finish");
    int check_value = rects[0][0];
    int index = 0;
    while(check_value != -1){
        for (size_t i = 0; i < 4; i++)
        {
            printf("%d,", rects[index][i]);
        }
        printf("\n");
        check_value = rects[++index][0];
    }

    free(array);

    return 0;
}