//
// Created by 仲凯宁 on 2019-11-15.
//

#ifndef INC_3150_A4_UTILS_H
#define INC_3150_A4_UTILS_H

#include "file_system.h"
//#include <cuda.h>
//#include <cuda_runtime.h>

__device__ void rename(FCB *target_fcb, const char *new_name) { // rename file
    int i = 0;
    while (*(new_name + i) != '\0') {
        target_fcb->name[i] = *(new_name + i);
        i++;
    }
}

__device__ int find_hole(FileSystem *fs, int blocks_needed) {
    // need to find blocks_needed number of available blocks
    int continuous = 1;

    for (int i = 0; i < 32768; i++) {
        if (fs->blocks[i].content[0] == '\0') { // the first empty block
            // check whether continuous
            for (int j = 0; j < 32; j++) {
                if (fs->blocks[i + j].content[0] != '\0') {
                    continuous = 0; // not continuous
                    break;
                }
            }

            if (continuous == 1) return i; // stop searching and return
            else continuous = 1; // continue searching
        }
    }
    printf("Fuck, no continuous space for your shit, too sad!\n");
}

__device__ void to_bitmap(unsigned int superblock[1024], int *bitmap) { // convert a int array to a bit array
    for (int i = 0; i < 1024; i++) {
        unsigned int num = superblock[i]; // this integer is converted to 32 bits
        for (int j = 31; j >= 0; j--) {
            if (num & (1 << j))
                bitmap[31 + i * 32 - j] = 1;
            else
                bitmap[31 + i * 32 - j] = 0;
        }
    }
}

__device__ void update_bitmap(int *bitmap, int block_address, int block_needed, int original_blocks_needed) {
    // blocks still need to use, set index to 1
    for (int i = 0; i < block_needed; i++) {
        bitmap[block_address + i] = 1;
    }

    // blocks no longer need to use, set index to 0
    for (int i = 0; i < original_blocks_needed - block_needed; i++) {
        bitmap[block_address + block_needed + i] = 0;
    }
}

__device__ void wipe_bitmap(int *bitmap, int block_address, int block_num) {
    for (int i = 0; i < block_num; i++) {
        bitmap[block_address + i] = 0;
    }
}

__device__ int pow(int base, int power) {
	int result = 1;
	for (int i = 0; i < power; i++) result *= base;
	return result;
}

__device__ void from_bitmap(unsigned int superblock[1024], int *bitmap) {
    for (int i = 0; i < 1024; i++) {
        unsigned int num = 0;
        for (int j = 0; j < 32; j++) {
            num += *(bitmap + 31 + i * 32 - j) * pow(2, j);
        }
        superblock[i] = num;
    }
}

__device__ bool strcompare(const char* src, const char* dst) {
	int ret = 0;
	while (!(ret = *(unsigned char*)src - *(unsigned char*)dst) && *dst) {
		src++;
		dst++;
	}
	if (ret < 0) ret = -1;
	else if (ret > 0) ret = 1;
	return ret;
}

__device__ char *strcopy(char *dst, const char *src) {
	int pos = 0;
	while (src[pos] != '\0') {
		dst[pos] = src[pos];
		pos++;
	}
}

__device__ char *strconcat(const char *str1, const char *str2) {
	int i = 0;
	char *new_str = new char[100];
	while (str1[i] != '\0') {
		new_str[i] = str1[i];
		i++;
	}

	int j = 0;
	while (str2[j] != '\0') {
		new_str[i + j] = str2[j];
		j++;
	}

	return new_str;
}

__device__ void copy_fcbs(const FCB *original_fcbs, FCB *new_fcbs, int valid_FCBs) {
	for (int i = 0; i < valid_FCBs; i++) {
		strcopy(new_fcbs[i].name, original_fcbs[i].name);
		new_fcbs[i].address = original_fcbs[i].address;
		new_fcbs[i].size = original_fcbs[i].size;
		new_fcbs[i].time = original_fcbs[i].time;
	}
}

__device__ void copy_nodes(const FS_node *original_nodes, FS_node *new_nodes, int valid_nodes) {
	for (int i = 0; i < valid_nodes; i++) {
		new_nodes[i].path = (char *) malloc(500);
		strcopy(new_nodes[i].path, original_nodes[i].path);
		new_nodes[i].size = original_nodes[i].size;
		new_nodes[i].time = original_nodes[i].time;
	}
}


__device__ void swap_FCB(FCB *fcb1, FCB *fcb2) {
    FCB temp = *fcb1;
    *fcb1 = *fcb2;
    *fcb2 = temp;
}

__device__ void swap_node(FS_node *node1, FS_node *node2) {
	FS_node temp = *node1;
	*node1 = *node2;
	*node2 = temp;
}

__device__ void sort_FCBs_byDate(FCB *sorted_fcbs, int len) {
    // sort FCBs using bubble sort
    for (int i = 0; i < len; i++) {
        for (int j = 1; j < len; j++) {
            if (sorted_fcbs[j - 1].time < sorted_fcbs[j].time)
                swap_FCB(&sorted_fcbs[j - 1], &sorted_fcbs[j]);
        }
    }
}

__device__ void sort_nodes_byDate(FS_node *sorted_nodes, int len) {
	// sort nodess using bubble sort
	for (int i = 0; i < len; i++) {
		for (int j = 1; j < len; j++) {
			if (sorted_nodes[j - 1].time < sorted_nodes[j].time)
				swap_node(&sorted_nodes[j - 1], &sorted_nodes[j]);
		}
	}
}

__device__ void sort_FCBs_bySize(FCB *sorted_fcbs, int len) {
    // sort FCBs using bubble sort
    for (int i = 0; i < len; i++) {
        for (int j = 1; j < len; j++) {
            if (sorted_fcbs[j - 1].size < sorted_fcbs[j].size)
                swap_FCB(&sorted_fcbs[j - 1], &sorted_fcbs[j]);
        }
    }
}

__device__ void sort_nodes_bySize(FS_node *sorted_nodes, int len) {
	// sort nodess using bubble sort
	for (int i = 0; i < len; i++) {
		for (int j = 1; j < len; j++) {
			if (sorted_nodes[j - 1].size < sorted_nodes[j].size)
				swap_node(&sorted_nodes[j - 1], &sorted_nodes[j]);
		}
	}
}

__device__ void print_fcb_byDate(FCB *sorted_fcbs, int len) {
    for (int i = 0; i < len; i++) {
        printf("File name: %s; modified time: %d\n", sorted_fcbs[i].name, sorted_fcbs[i].time);
    }
}

__device__ void print_FS_byDate(FS_node *current_node, int childs) {
	for (int i = 0; i < childs; i++) {
		if (!strcompare(current_node->child_node[i].path, "")) continue;
		printf("Folder name: %s; modified time: %d\n", current_node->child_node[i].path, current_node->child_node[i].time);
	}
}

__device__ void print_fcb_bySize(FCB *sorted_fcbs, int len) {
    for (int i = 0; i < len; i++) {
        printf("File name: %s; file size: %d\n", sorted_fcbs[i].name, sorted_fcbs[i].size);
    }
}

__device__ void print_FS_bySize(FS_node *current_node, int childs) {
	for (int i = 0; i < childs; i++) {
		if (!strcompare(current_node->child_node[i].path, "")) continue;
		printf("Folder name: %s; folder size: %d\n", current_node->child_node[i].path,
			current_node->child_node[i].size);
	}
}


__device__ void rearrange_fcbs(FCB *fcbs, int empty_idx){
    int last_valid;

    // find last non-empty fcb
    int current_pos = empty_idx+1;
    while (fcbs[current_pos].address != -1) current_pos++;
    last_valid = current_pos-1;

    // place this fcb to the empty position
    swap_FCB(&fcbs[empty_idx], &fcbs[last_valid]);
}

__device__ char *get_abs_path(char *prefix, char *rela_path) {
	char *prefix0 = (char *)malloc(50);
	strcopy(prefix0, prefix);
	char *prefix1 = strconcat(prefix0, "/");
	char *abs_path = strconcat(prefix1, rela_path);
	return abs_path;
}

__device__ int name_size(const char *s) { // note: I don't count '\0' as a character
	int len = 0;
	while (s[len] != '\0') len++;
	return len;
}

#endif //INC_3150_A4_UTILS_H
