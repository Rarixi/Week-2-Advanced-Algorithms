# Define a function to perform bubble sort on a list of numbers
def bubble_sort(arr: list[int]) -> list[int]:
    # Get length of the array
    n = len(arr)
    # Track whether the array is sorted or not
    sorted = False
    # Repeat when sorted is False
    while sorted == False:
        # Set sorted to True at the beginning of each pass
        sorted = True
        #For each index i in the array, starting at 0 and stopping before the last index
        for i in range(0, n - 1):
            #Compare the current element with the next element
            if arr[i] > arr[i + 1]:
                # Swap the elements
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                # Set sorted to False since a swap occurred
                sorted = False
    # Return the sorted array
    return arr










def selection_sort(arr: list[int]) -> list[int]:
    # Get the length of the array
    n = len(arr)
    # For each index i in the array, starting at 0 and stopping before the last index
    for i in range(n):
        # Assume the minimum is the first element of the unsorted part
        min_index = i
        # For each index j in the unsorted part of the array, starting at i + 1
        for j in range(i + 1, n):
            # If the current element is less than the assumed minimum, update min_index
            if arr[j] < arr[min_index]:
                min_index = j
        # Swap the found minimum element with the first element of the unsorted part
        arr[i], arr[min_index] = arr[min_index], arr[i]
    # Return the sorted array
    return arr



def insertion_sort(arr: list[int]) -> list[int]:
    # Get the length of the array
    n = len(arr)
    # For each index i in the array, starting at 1 and stopping before the last index
    for i in range(1, n):
        # Store the current element to be inserted
        key = arr[i]
        # Initialize j to the index of the last sorted element
        j = i - 1
        # Move elements of arr[0..i-1], that are greater than key, to one position ahead of their current position
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        # Insert the key at its correct position
        arr[j + 1] = key
    # Return the sorted array
    return arr

