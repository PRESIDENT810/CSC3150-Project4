#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

#define G_WRITE 1
#define G_READ 2
#define LS_D 3
#define LS_S 4
#define RM 5
#define CD 6
#define CD_P 7
#define PWD 8
#define RM_RF 9
#define MKDIR 10

struct FCB {
	char name[20]; // 20 bytes to store name of the file
	int address = -1; // 4 byte to store block address of the file
	int size; // 4 byte to store size of the file (blocks occcupied)
	int time; // 4 byte to store time of the file
};

struct Block { // each Block takes 32 bytes
	char content[32] = { '\0' };
};

struct FS_node {
	FS_node *parent_node = (FS_node *)malloc(1000);
	FS_node *child_node = (FS_node *)malloc(1000);
	int childs = 0;
	struct FileSystem *this_FS = (FileSystem *)malloc(1085000);
	char *path = (char *)malloc(50 * sizeof(char));

	int size = 0;
	int time = 0;
};

struct FileSystem { // 1085440+36=1085476 byte
	// 32+1024+4=1060KB (1085440 byte)
	FCB *fcbs = (FCB *)malloc(1024 * 32); // 1024 FCBs, 32KB in total
	Block *blocks = (Block *)malloc(100 * 32); // 32768 Blocks, each Block takes 32 bytes, 1024KB in total
	unsigned int SuperBlock[1024]; // (converted to bitmap) 32768 bits indicating whether a block is occupied, 4KB in total

	FS_node *node_ptr = NULL; // pointer to the node containing this FS

	// 4*9=36byte
	int SUPERBLOCK_SIZE;
	int FCB_SIZE;
	int FCB_ENTRIES;
	int STORAGE_SIZE;
	int STORAGE_BLOCK_SIZE;
	int MAX_FILENAME_SIZE;
	int MAX_FILE_NUM;
	int MAX_FILE_SIZE;
	int FILE_BASE_ADDRESS;
	int gtime;
};


__device__ void fs_init(FileSystem *fs, int SUPERBLOCK_SIZE,
             int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
             int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
             int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS);

__device__ u32 fs_open(FileSystem *fs, char *s, int op);

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);

__device__ u32 fs_write(FileSystem *fs, uchar *input, u32 size, u32 fp);

__device__ void fs_gsys(FileSystem *fs, int op);

__device__ void fs_gsys(FileSystem *fs, int op, char *s);


#endif
