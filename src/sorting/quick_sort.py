import random

#Create a function called quick_sort that takes a list of numbers, and returns a sorted list of numbers
def quick_sort(arr: list[int]) -> list[int]:
    #Makes copy of the original list to avoid modifying it
    result = arr.copy()
    #If the list has 1 or fewer elements, it is sorted, so return a copy of the list
    if len(result) <= 1:
        return result

    #Run quick sort on the whole list, starting from first index (0) to the last index (len(result) - 1)
    _quick_sort(result, 0, len(result) - 1)
    return result

#Create a helper function called _quick_sort that sorts one section of the list, starting from the low index to the high index
def _quick_sort(arr: list[int], low: int, high: int) -> None:

    #While the low index is less than the high index, it means there are at least two elements to sort
    while low < high:
        #Count how many numbers are in this section of the list, and if there are 10 or fewer numbers, use insertion sort instead of quick sort
        if high - low + 1 <= 10:

            #Use insertion sort on this section of the list, starting from the low index to the high index
            insertion_sort(arr, low, high)
            return

        #Choose a random pivot , and rearrange the numbers around that pivot
        pivot_index = partition(arr, low, high)

        if pivot_index - low < high - pivot_index:
            #Quick sort the left sections of the list, excluding the pivot
            _quick_sort(arr, low, pivot_index - 1)
            low = pivot_index + 1
        else:
        
            #Quick sort the right section of the list, excluding the pivot
            _quick_sort(arr, pivot_index + 1, high)
            high = pivot_index - 1

#Create a function called partition , that rearranges the numbers in a section of the list around a pivot, and returns the index of the pivot after rearranging
def partition(arr: list[int], low: int, high: int) -> int:

    #Choose a random index between low and high, inclusive
    random_index = random.randint(low, high)

    # Move random pivot to the end
    arr[random_index], arr[high] = arr[high], arr[random_index]

    #Set the last element as the pivot
    pivot = arr[high]

    #Initialize the index of the smaller element
    i = low - 1

    #For each index j in the section of the list, starting from low and stopping before high or before the pivot
    for j in range(low, high):
        #If the current element is less than or equal to the pivot, swap it with the element at index i + 1
        if arr[j] <= pivot:
            i += 1
            #Move the smaller element to the left side
            arr[i], arr[j] = arr[j], arr[i]

    #Move the pivot from the end to its correct position, which is index i + 1
    arr[i + 1], arr[high] = arr[high], arr[i + 1]

    #Return the index of the pivot after rearranging
    return i + 1

#Create a function called insertion_sort that sorts a section of the list, starting from the low index to the high index
def insertion_sort(arr: list[int], low: int, high: int) -> None:

    #For each index i in the section of the list, starting from low + 1 and stopping before high + 1
    for i in range(low + 1, high + 1):

        #Store the current element to be inserted
        key = arr[i]
        #Initialize j to the index of the last sorted element
        j = i - 1
        #While J is still inside the section and the number at index j is greater than the key, move the number at index j to the right
        while j >= low and arr[j] > key:
            arr[j + 1] = arr[j]
            #Move one index to the left
            j -= 1
        #Insert the key at its correct position, which is index j + 1
        arr[j + 1] = key
