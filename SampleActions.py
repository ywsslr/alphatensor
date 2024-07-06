import numpy as np

def generate_move(size, F):
    """
    Generates three arrays of the given size with elements in F such as {-1, 0, 1}.
    
    The first vector's first non-zero element is guaranteed to be positive.

    Args:
        size (int): The size of the arrays to generate.
        F (iterable): the element of arrays to generate 

    Returns:
        tuple: A tuple containing three numpy arrays of the given size.
    """
    def generate_non_zero_vector():
        while True:
            vector = np.random.choice(F, size)
            if np.any(vector != 0):
                return vector

    def generate_first_vector():
        vector = generate_non_zero_vector()
        for i in range(size):
            if vector[i] != 0:
                vector[i] = abs(vector[i])
                break
        return vector

    vector1 = generate_first_vector()
    vector2 = generate_non_zero_vector()
    vector3 = generate_non_zero_vector()

    return vector1, vector2, vector3

def generate_unique_samples(size, num_samples, F):
    """
    Generates a specified number of unique triplets of vectors using `generate_move`.

    Parameters
    ----------
    size : int
        The size of each vector.
    num_samples : int
        The number of unique triplets to generate.

    Returns
    -------
    list of tuple of numpy.ndarray
        A list containing unique tuples of three numpy arrays.
    """
    samples = set()

    while len(samples) < num_samples:
        sample = generate_move(size, F)
        sample_tuple = tuple(map(tuple, sample))  # Convert each array to a tuple to make it hashable
        if sample_tuple not in samples:
            samples.add(sample_tuple)

    # Convert set of tuples back to list of numpy arrays
    return [tuple(map(np.array, sample)) for sample in samples]

if __name__ == "__main__":
    ## test 
    size = 4
    F = (-1,0,1)
    # # generate_move
    # v1, v2, v3 = generate_move(size,F)
    # print("Vector 1:", v1)
    # print("Vector 2:", v2)
    # print("Vector 3:", v3, type(v3))
    

    # generate_unique_samples
    num_samples = 100
    unique_samples = generate_unique_samples(size, num_samples, F)
    for i, sample in enumerate(unique_samples):
        print(f"Sample {i + 1}:")
        print("Vector 1:", sample[0])
        print("Vector 2:", sample[1])
        print("Vector 3:", sample[2])
        print()
    print(type(unique_samples[-1][0]))

