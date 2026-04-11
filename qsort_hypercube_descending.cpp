// -----------------------------------------------------------------------------
// Hypercube Quicksort to sort a list of integers in DESCENDING ORDER
// distributed across processors. MPI-based implementation.
//
// Changes from ascending version:
//   1. compare_int returns reversed comparison (descending qsort)
//   2. split_list_index finds smallest index where list[idx] < pivot
//      (i.e., list[0..idx-1] >= pivot, list[idx..end] < pivot)
//   3. merged_list merges in descending order
//   4. Lower-ranked process keeps elements >= pivot (the larger values)
//      Higher-ranked process keeps elements < pivot (the smaller values)
//   5. check_list verifies descending order globally
//
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <new>
#include <mpi.h>

#define MAX_LIST_SIZE_PER_PROC	268435456

#ifndef VERBOSE
#define VERBOSE 0			// Use VERBOSE to control output 
#endif

int compare_int_desc(const void *, const void *);

class HyperCube_Class {
    public:
	void Initialize(int, int, int);
	void HyperCube_QuickSort();
	void print_list();
	void check_list();

	int dimension;		// Hypercube dimension

	int num_procs; 		// Number of MPI processes
	int my_id;		// Rank/id of this process

	int list_size;		// Local list size
	int list_size_initial;	// Size of initial local list (before sorting)

    private:
	int * initialize_list(int);
	int neighbor_along_dim_k(int );
	int * merged_list(int *, int, int *, int); 
	int split_list_index (int *, int, int); 
	void print_local_list();

	int *list;		// Local list
};

//
// Initialize HyperCube
//
void HyperCube_Class::Initialize(int dim, int size, int type) {

    dimension = dim; 		// Hypercube dimension

    MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_id);

    list_size_initial = size;
    list_size = list_size_initial; 

    list = initialize_list(type);
}

int HyperCube_Class::neighbor_along_dim_k(int k) {
    int mask = 1 << (k-1); 
    return (my_id ^ mask); 
}

// Merge two sorted lists in DESCENDING order and return merged list
//
int * HyperCube_Class::merged_list(int * list1, int list1_size, int * list2, int list2_size) {
    int * list = new int[list1_size+list2_size];
    int idx1 = 0; 
    int idx2 = 0; 
    int idx = 0; 
    while ((idx1 < list1_size) && (idx2 < list2_size)) {
	// Descending: take the larger element first
	if (list1[idx1] >= list2[idx2]) {
	    list[idx] = list1[idx1]; 
	    idx++; idx1++;
	} else {
	    list[idx] = list2[idx2]; 
	    idx++; idx2++;
	}
    }
    while (idx1 < list1_size) {
	list[idx] = list1[idx1]; 
	idx++; idx1++;
    }
    while (idx2 < list2_size) {
	list[idx] = list2[idx2]; 
	idx++; idx2++;
    }
    return list;
}

// For descending sort: find the first index where list[idx] < pivot.
// After the call:
//   list[0 ... idx-1] >= pivot   (these are the "large" elements)
//   list[idx ... list_size-1] < pivot  (these are the "small" elements)
// List is sorted in descending order, so we binary-search for the
// transition from >= pivot to < pivot.
//
int HyperCube_Class::split_list_index (int *list, int list_size, int pivot) {
    int first, last, mid;  
    first = 0; last = list_size; mid = (first+last)/2;
    while (first < last) {
	// Descending list: list[mid] >= pivot means we haven't crossed yet
	if (list[mid] >= pivot) {
	    first = mid+1; mid = (first+last)/2;
	} else {
	    last = mid; mid = (first+last)/2;
	}
    }
    return last;
}

void HyperCube_Class::print_local_list() {
    int j;
    for (j = 0; j < list_size; j++) {
	if ((j % 8) == 0) printf("[Proc: %0d]", my_id);
	printf(" %8d", list[j]); 
	if ((j % 8) == 7) printf("\n"); 
    }
    printf("\n"); 
    return;
}

void HyperCube_Class::print_list() {
    int tag = 0;
    int dummy = 0;
    MPI_Status status; 
    if (my_id-1 >= 0) {
	MPI_Recv(&dummy, 1, MPI_INT, my_id-1, tag, MPI_COMM_WORLD, &status);
    }
    print_local_list(); 
    if (my_id+1 < num_procs) {
	MPI_Send(&dummy, 1, MPI_INT, my_id+1, tag, MPI_COMM_WORLD);
    }
}

int * HyperCube_Class::initialize_list(int type) {
    int j;
    int * list = new int[list_size]; 
    switch (type) {
	case -1:	// Elements are in descending order
	    for (j = 0; j < list_size; j++) {
		list[j] = (num_procs-my_id)*list_size-j;
	    }
	    break;
	case -2:	// Elements are in ascending order
	    for (j = 0; j < list_size; j++) {
		list[j] = my_id*list_size+j+1;
	    }
	    break;
	default: 
	    srand48(type + my_id); 
	    list[0] = lrand48() % 100;
	    for (j = 1; j < list_size; j++) {
		list[j] = lrand48() % 100;
	    }
	    break;
    }
    return list;
}

// Check if list is sorted in DESCENDING order globally.
// Process P0 has the largest values; process P(p-1) has the smallest.
// Each process checks:
//   (a) its local list is non-increasing
//   (b) its first element is <= the last element of the previous process
//       (the previous process holds larger values)
//
void HyperCube_Class::check_list() {
    int tag = 0;
    int min_nbr = INT_MAX;	// Minimum from the previous (lower-ranked) process
    int error, local_error;
    int j, my_min;
    MPI_Status status; 

    // Receive the smallest value from process (my_id-1).
    // In descending global order, process (my_id-1) holds larger values,
    // so its minimum must be >= our maximum (list[0]).
    if (my_id-1 >= 0) {
	MPI_Recv(&min_nbr, 1, MPI_INT, my_id-1, tag, MPI_COMM_WORLD, &status);
    }

    local_error = 0;
    if (list_size > 0) {
	// Our largest element (list[0]) must be <= the smallest on the previous process
	if (list[0] > min_nbr) local_error = 1;
	// Check local list is non-increasing
	for (j = 1; j < list_size; j++) {
	    if (list[j] > list[j-1]) local_error = 1;
	}
	my_min = list[list_size-1];
    } else {
	my_min = min_nbr;
    }

    if (VERBOSE > 1) {
	printf("[Proc: %0d] check_list: local_error = %d\n", my_id, local_error);
    }

    // Send our minimum to process (my_id+1)
    if (my_id+1 < num_procs) {
	MPI_Send(&my_min, 1, MPI_INT, my_id+1, tag, MPI_COMM_WORLD);
    }

    MPI_Reduce(&local_error, &error, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (my_id == 0) {
	if (error == 0) {
	    printf("[Proc: %0d] Congratulations. The list has been sorted correctly.\n", my_id);
	} else {
	    printf("[Proc: %0d] Error encountered. The list has not been sorted correctly.\n", my_id);
	}
    }
}

//
// HyperCube Quicksort (Descending)
//
// In the descending case the roles of "low" and "high" are swapped:
//   - Lower-ranked processes keep elements >= pivot (the larger half)
//   - Higher-ranked processes keep elements <  pivot (the smaller half)
//
void HyperCube_Class::HyperCube_QuickSort() {
    int k; 
    int nbr_k; 

    int local_median;
    int pivot;
    int idx;
    // After split_list_index (descending version):
    //   list[0 ... idx-1] >= pivot   (geq portion)
    //   list[idx ... list_size-1] < pivot  (lt portion)
    int list_size_geq;		// Number of elements >= pivot
    int list_size_lt;		// Number of elements < pivot
    int * nbr_list; 
    int nbr_list_size;
    int * new_list; 
    int tag = 0;
    MPI_Status status; 

    int error, mask, i;

    int sub_hypercube_size; 
    int * sub_hypercube_processors;
    MPI_Group sub_hypercube_group; 
    MPI_Comm sub_hypercube_comm;
    MPI_Group hypercube_group;

    // Sort local list in descending order
    qsort(list, list_size, sizeof(int), compare_int_desc);

    MPI_Comm_group(MPI_COMM_WORLD, &hypercube_group);

    for (k = dimension; k > 0; k--) {

	sub_hypercube_size = (int) pow(2,k); mask = (~0) << k;
	sub_hypercube_processors = new int[sub_hypercube_size]; 
	sub_hypercube_processors[0] = my_id & mask; 
	for (i = 1; i < sub_hypercube_size; i++) {
	    sub_hypercube_processors[i] = sub_hypercube_processors[i-1]+1;
	}

	MPI_Group_incl(hypercube_group, sub_hypercube_size, sub_hypercube_processors, &sub_hypercube_group);
	MPI_Comm_create(MPI_COMM_WORLD, sub_hypercube_group, &sub_hypercube_comm);

	// Median of locally sorted (descending) list
	local_median = list[list_size/2];

	// MPI-1: Compute pivot = mean of medians across sub-hypercube
	MPI_Allreduce(&local_median, &pivot, 1, MPI_INT, MPI_SUM, sub_hypercube_comm);

	pivot = pivot/sub_hypercube_size;

	// Split: list[0..idx-1] >= pivot, list[idx..end] < pivot
	idx = split_list_index(list, list_size, pivot);

	list_size_geq = idx;
	list_size_lt  = list_size - idx;

	nbr_k = neighbor_along_dim_k(k); 

	if (nbr_k > my_id) {
	    // Lower-ranked process: keeps elements >= pivot (the larger values).
	    // Sends its "< pivot" portion to higher-ranked neighbor,
	    // receives neighbor's ">= pivot" portion in return.

	    // MPI-2: Send count of elements < pivot to higher-ranked neighbor
	    MPI_Send(&list_size_lt, 1, MPI_INT, nbr_k, tag, MPI_COMM_WORLD);

	    // MPI-3: Receive count of elements >= pivot from higher-ranked neighbor
	    MPI_Recv(&nbr_list_size, 1, MPI_INT, nbr_k, tag, MPI_COMM_WORLD, &status);

	    nbr_list = new int[nbr_list_size];

	    // MPI-4: Send list[idx...end] (elements < pivot) to higher-ranked neighbor
	    MPI_Send(&list[idx], list_size_lt, MPI_INT, nbr_k, tag, MPI_COMM_WORLD);

	    // MPI-5: Receive neighbor's elements >= pivot
	    MPI_Recv(nbr_list, nbr_list_size, MPI_INT, nbr_k, tag, MPI_COMM_WORLD, &status);

	    // Merge local >= pivot portion with neighbor's >= pivot portion (descending)
	    new_list = merged_list(list, idx, nbr_list, nbr_list_size); 

	    delete [] list; 
	    delete [] nbr_list; 
	    list = new_list; 
	    list_size = list_size_geq + nbr_list_size;

	} else {
	    // Higher-ranked process: keeps elements < pivot (the smaller values).
	    // Receives "< pivot" count from lower-ranked neighbor,
	    // sends its own ">= pivot" count.

	    // MPI-6: Receive count of elements < pivot from lower-ranked neighbor
	    MPI_Recv(&nbr_list_size, 1, MPI_INT, nbr_k, tag, MPI_COMM_WORLD, &status);

	    // MPI-7: Send count of elements >= pivot to lower-ranked neighbor
	    MPI_Send(&list_size_geq, 1, MPI_INT, nbr_k, tag, MPI_COMM_WORLD);

	    nbr_list = new int[nbr_list_size]; 

	    // MPI-8: Receive neighbor's elements < pivot
	    MPI_Recv(nbr_list, nbr_list_size, MPI_INT, nbr_k, tag, MPI_COMM_WORLD, &status);

	    // MPI-9: Send list[0...idx-1] (elements >= pivot) to lower-ranked neighbor
	    MPI_Send(&list[0], list_size_geq, MPI_INT, nbr_k, tag, MPI_COMM_WORLD);

	    // Merge local < pivot portion with neighbor's < pivot portion (descending)
	    new_list = merged_list(&list[idx], list_size_lt, nbr_list, nbr_list_size); 

	    delete [] list; 
	    delete [] nbr_list; 
	    list = new_list; 
	    list_size = list_size_lt + nbr_list_size;
	}

	MPI_Group_free(&sub_hypercube_group);
	MPI_Comm_free(&sub_hypercube_comm);
	delete [] sub_hypercube_processors;
    }
}

// Comparison for descending qsort
int compare_int_desc(const void *a0, const void *b0) {
    int a = *(int *)a0;
    int b = *(int *)b0;
    if (a > b) {
	return -1; 
    } else if (a < b) {
	return 1;
    } else {
	return 0;
    }
}

//------------------------------------------------------------------------------
// Main program
// 
int main(int argc, char *argv[])
{
    int size; 
    int type;
    int dim;

    int my_id; 
    int num_procs;
    double start, total_time;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_id);

    if (argc != 3)  {
	if (my_id == 0) 
	    printf("Usage: mpirun -np <number_of_processes> <executable_name> <list_size_per_process> <type>\n");
	exit(0);
    }

    size = atoi(argv[argc-2]);
    if ((size <= 0) || (size > MAX_LIST_SIZE_PER_PROC)) {
	if (my_id == 0)
	    printf("List size outside range [%d ... %d]. Aborting ...\n", 1, MAX_LIST_SIZE_PER_PROC);
	exit(0);
    }

    type = atoi(argv[argc-1]);

    dim = (int) log2((double) num_procs); 
    if (num_procs != (int) pow(2,dim)) {
	if (my_id == 0) 
	    printf("Number of processors must be power of 2. Aborting ...\n"); 
	exit(0); 
    }

    HyperCube_Class HyperCube;
    HyperCube.Initialize(dim, size, type);

    if (VERBOSE > 2) {
	HyperCube.print_list();
    }

    start = MPI_Wtime(); 
    HyperCube.HyperCube_QuickSort(); 
    total_time = MPI_Wtime()-start;

    if (my_id == 0) {
	printf("[Proc: %0d] number of processes = %d, ", HyperCube.my_id, HyperCube.num_procs);
	printf("initial local list size = %d, ", HyperCube.list_size_initial);
	printf("hypercube quicksort time = %f\n", total_time); 
    }

    HyperCube.check_list();

    if (VERBOSE > 2) {
	HyperCube.print_list();
    }

    MPI_Finalize();
}
