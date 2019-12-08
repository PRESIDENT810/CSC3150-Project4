#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.h"

__device__ __managed__ u32 gtime = 0;


__device__ void fs_init(FileSystem *fs, int SUPERBLOCK_SIZE,
						int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
						int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
						int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS) {

	// init variables (already initiated in struct declaration)
	for (int i = 0; i < 1024; i++) {
		fs->fcbs[i].address = -1;
		fs->fcbs[i].time = 0;
		fs->fcbs[i].size = 0;
	}

	// init constants
	fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
	fs->FCB_SIZE = FCB_SIZE;
	fs->FCB_ENTRIES = FCB_ENTRIES;
	fs->STORAGE_SIZE = VOLUME_SIZE;
	fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
	fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
	fs->MAX_FILE_NUM = MAX_FILE_NUM;
	fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
	fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;
	fs->gtime = 0;
}

__device__ void fs_init(FileSystem *fs) {

	// init variables (already initiated in struct declaration)
	for (int i = 0; i < 1024; i++) {
		fs->fcbs[i].address = -1;
		fs->fcbs[i].time = 0;
		fs->fcbs[i].size = 0;
	}

	fs->blocks = (Block *)malloc(1095000);

	// init constants
	fs->gtime = 0;
}


__device__ u32 fs_open(FileSystem *fs, char *s, int op) {
	fs->gtime++;
	/* Implement open operation here */
	int found = 0;
	u32 fcb_idx = -1;
	FCB *target_fcb = new FCB;
	
	// find fbc via filename
	for (int i = 0; i < 1024; i++) {
		if (!strcompare(fs->fcbs[i].name, s)) {
			target_fcb = &(fs->fcbs[i]);
			fcb_idx = i;
			found = 1;
			break;
		}
	}
	// if file not found in the file system when write: create a new FCB and return its index
	if (found == 0 && op == G_WRITE) {
		fs->node_ptr->size += name_size(s);

		rename(target_fcb, s);
		target_fcb->address = find_hole(fs,1); // allocate each first write 1 block (don't know whether this hole is big enough)
		

		target_fcb->size = 0; // this is a 0-byte file
		target_fcb->time = fs->gtime; // record current time

		// find a empty fcb to store this new file info
		int empty_idx = -1;
		for (int i = 0; i < 1024; i++) {
			if (fs->fcbs[i].address == -1) {
				empty_idx = i;
				break;
			}
		}

		// if no empty fcb, then file number exceeds 1024
		if (empty_idx == -1) {
			printf("Too much file to handle!\n");
		}

		// assign target fcb to this empty fcb and return
		fs->fcbs[empty_idx] = *(target_fcb);
		return empty_idx;
	}

	// if file not found in the file system when read: error
	else if (found == 0 && op == G_READ) {
		printf("File not found!\n");
	}

	// if file found in the file system: return the index of FCB
	else {
		return fcb_idx;
	}
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp) {
	fs->gtime++;
	/* Implement read operation here */
	FCB *target_fcb = &(fs->fcbs[fp]);
	target_fcb->time = fs->gtime;
	int blocks_needed = (size - 1) / 32 + 1;

	//    if (blocks_needed>target_fcb->size){ // try to visit address beyond target file's range
	//        printf("Segmentation fault");
	//        exit(1);
	//    }

	int start_address = target_fcb->address;

	// read blocks
	for (int block = 0; block < blocks_needed; block++) {
		// read each block
		for (int i = 0; i < 32; i++) {
			int current_pos = block * 32 + i;
			char character = fs->blocks[block + start_address].content[i];
			*(output + current_pos) = character;
			if (character == '\0') break;
		}
	}
}

__device__ u32 fs_write(FileSystem *fs, uchar *input, u32 size, u32 fp) {
	fs->gtime++;
	/* Implement write operation here */
	FCB *target_fcb = &(fs->fcbs[fp]);
	int blocks_needed = (size - 1) / 32 + 1;
	int original_blocks_needed = (target_fcb->size - 1) / 32 + 1;
	target_fcb->time = fs->gtime;

	target_fcb->size = size;
	//    if (size != -1) { // if not first write
	//        int exceed = 0;
	//        if (old_size > size) { // space is enough
	//            target_fcb->size = blocks_needed;
	//        } else { // space is not enough
	//            blocks_needed = old_size;
	//            exceed = 1;
	//        }
	//    }

		// find a hole enough to place the input
	//    int start_address = find_hole(fs, blocks_needed);
	//    target_fcb->address = start_address;
	int start_address = target_fcb->address;


	// clean up blocks & write content into blocks
	for (int block = 0; block < blocks_needed; block++) {
		// clean each block
		for (int i = 0; i < 32; i++) {
			fs->blocks[block + start_address].content[i] = '\0';
		}
		// write each block
		for (int i = 0; i < 32; i++) {
			int current_pos = block * 32 + i;
			char character = *(input + current_pos);
			if (character == '\0') break;
			fs->blocks[block + start_address].content[i] = character;
		}
	}

	// update bitmap
	int bitmap[32768] = { 0 };
	to_bitmap(fs->SuperBlock, bitmap);
	update_bitmap(bitmap, start_address, blocks_needed, original_blocks_needed);
	from_bitmap(fs->SuperBlock, bitmap);

	return 0;
}

__device__ void fs_gsys(FileSystem *fs, int op) { // list all files sorted by date/size
	if (op == LS_S || op == LS_D) {
		fs->gtime++;
		/* Implement LS_D and LS_S operation here */

		// count how many FCB used
		int valid_FCBs = 0;
		while (fs->fcbs[valid_FCBs].address != -1 && valid_FCBs != 1023) valid_FCBs++;

		FCB *sorted_fcbs = (FCB *) malloc(valid_FCBs*400);
		FS_node *sorted_nodes = (FS_node *) malloc(1000*fs->node_ptr->childs);

		// copy valid fcbs to sorted fcbs
		copy_fcbs(fs->fcbs, sorted_fcbs, valid_FCBs);
		copy_nodes(fs->node_ptr->child_node, sorted_nodes, fs->node_ptr->childs);

		int childs = fs->node_ptr->childs;

		// sort fcbs
		if (op == LS_D) {
			printf("===Sort by modified time===\n");
			sort_FCBs_byDate(sorted_fcbs, valid_FCBs);
			sort_nodes_byDate(sorted_nodes, fs->node_ptr->childs);
			if (sorted_nodes[childs].time > sorted_fcbs[valid_FCBs].time) {
				print_FS_byDate(fs->node_ptr, childs);
				print_fcb_byDate(sorted_fcbs, valid_FCBs);
			}
			else {
				print_fcb_byDate(sorted_fcbs, valid_FCBs);
				print_FS_byDate(fs->node_ptr, childs);
			}
		}
		else {
			printf("===Sort by file size===\n");
			sort_FCBs_bySize(sorted_fcbs, valid_FCBs);
			sort_nodes_bySize(sorted_nodes, fs->node_ptr->childs);

			if (sorted_nodes[childs].size > sorted_fcbs[valid_FCBs].size) {
				print_FS_bySize(fs->node_ptr, childs);
				print_fcb_bySize(sorted_fcbs, valid_FCBs);
			}
			else {
				print_fcb_bySize(sorted_fcbs, valid_FCBs);
				print_FS_bySize(fs->node_ptr, childs);
			}
		}
	}

	else if (op == PWD) {
		printf("Current path: %s\n", fs->node_ptr->path);
	}

	else if (op == CD_P) {
		int old_size = fs->node_ptr->size;
		FileSystem *new_fs = new FileSystem;
		new_fs = fs->node_ptr->parent_node->this_FS;
		*fs = *new_fs;
		fs->node_ptr->child_node[0].size = old_size;
	}
}


__device__ void fs_gsys(FileSystem *fs, int op, char *s) {
	if (op == RM) {
		fs->gtime++;
		/* Implement rm operation here */
		int found = 0;
		u32 fcb_idx = -1;
		FCB *target_fcb = new FCB;

		// find fbc via filename
		for (int i = 0; i < 1024; i++) {
			if (!strcompare(fs->fcbs[i].name, s)) {
				target_fcb = &(fs->fcbs[i]);
				fcb_idx = i;
				found = 1;
				break;
			}
		}

		if (found == 1) { // file to be deleted found
			int address = target_fcb->address;
			int size = target_fcb->size;

			// wipe out bitmap
			int bitmap[32768] = { 0 };
			to_bitmap(fs->SuperBlock, bitmap);
			wipe_bitmap(bitmap, address, size);
			from_bitmap(fs->SuperBlock, bitmap);

			// wipe out FCB
			target_fcb->time = 0;
			target_fcb->size = 0;
			target_fcb->address = -1;
			strcopy(target_fcb->name, "\0");

			// rearange FCBs so no empty FCB before used FCB
			rearrange_fcbs(fs->fcbs, fcb_idx);
		}
		else {
			printf("No such file to delete!\n");
		}
	}

	else if (op == MKDIR) {
		fs->node_ptr->size += name_size(s);

		FS_node *sub_node = new FS_node;
		FileSystem *sub_fs = new FileSystem;


		fs_init(sub_fs);
		sub_fs->blocks = new Block[50];

		sub_fs->gtime = fs->gtime;

		// give abs path to sub node
		char *abs_path = get_abs_path(fs->node_ptr->path, s);
		strcopy(sub_node->path, abs_path);

		// link sub node and sub fs
		sub_fs->node_ptr = sub_node;
		sub_node->this_FS = sub_fs;
		sub_node->time = fs->gtime;
		sub_node->size = 0;
		fs->node_ptr->size += name_size(s);



		// link child and parent
		sub_node->parent_node = fs->node_ptr;
		fs->node_ptr->child_node[fs->node_ptr->childs] = *sub_node;
		fs->node_ptr->childs++;


	}

	else if (op == RM_RF) {
		fs->node_ptr->size -= name_size(s);
		// get abs path
		char *abs_path = get_abs_path(fs->node_ptr->path, s);

		// find the node
		int child_idx = 0;
		FS_node *current_node = fs->node_ptr;
		//while (!strcompare(current_node->child_node[child_idx].path, abs_path)) child_idx++;

		// use a empty node to replace this shit
		FS_node *empty_node = new FS_node;
		current_node->child_node[child_idx] = *empty_node;
	}

	else if (op == CD) {
		// get abs path
		char *abs_path = get_abs_path(fs->node_ptr->path, s);

		// find the node
		int child_idx = 0;
		FS_node *current_node = fs->node_ptr;
		//while (!strcompare(current_node->child_node[child_idx].path, abs_path)) child_idx++;

		FileSystem *new_fs = new FileSystem;
		FileSystem *old_fs = new FileSystem;

		fs_init(new_fs);
		fs_init(old_fs);

		*old_fs = *fs;
		*new_fs = *current_node->child_node[child_idx].this_FS;
		*fs = *new_fs;
		fs->node_ptr->parent_node->this_FS = old_fs;
	}
}