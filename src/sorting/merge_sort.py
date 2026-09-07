
# Create a function called merge_sort that takes list of numbers
def merge_sort(arr: list[int]) -> list[int]:

    # If the list has 1 or fewer elements, it is sorted, so return a copy of the list 
    if len(arr) <= 1:
        return arr.copy()
    #Finds the middle index of the list
    middle = len(arr) // 2

    #Take everything from the beginning to the middle, and sort left half
    left = merge_sort(arr[:middle])

    #Take everything from the middle to the end, and sort right half
    right = merge_sort(arr[middle:])

    return merge(left, right)

#Create a function called merge, give it two list of numbers, and return a single sorted list
def merge(left: list[int], right: list[int]) -> list[int]:
    #Create an empty list to hold sorted numbers
    result = []

    #Start at index 0 for both left and right lists
    i = 0
    j = 0

    #While there are still elements in both left and right lists
    while i < len(left) and j < len(right):

        #Compare the current elements from the left and right lists, and append the smaller one to the result list
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            #The right element is smaller, so append it to the result list and move to the next element in the right list
            result.append(right[j])
            j += 1

    #Once one of the lists is exhausted, append the remaining elements from the other list to the result list
    result.extend(left[i:])
    result.extend(right[j:])

    return result
