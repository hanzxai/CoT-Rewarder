"""
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和

例如：
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: The subarray [4,-1,2,1] has the largest sum 6.
"""


def maxSubArray(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    maxnums = nums[0]
    cursum = nums[0]

    for i in nums[1:]:
        cursum = max(i, cursum + i) # =1  1
        maxnums = max(maxnums, cursum) # =1  1
    return maxnums

nums = [-2,1,-3,4,-1,2,1,-5,4]
print(maxSubArray(nums))