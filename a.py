from collections import defaultdict


class Solution:
    def findOrder(self, numCourses: int, prerequisites):
        inDegree = [0]*numCourses
        adj = defaultdict(list)
        q = []
        for i,j in prerequisites:
            inDegree[i] += 1
            adj[j].append(i)
        for i in inDegree:
            if i == 0:
                q.append(i)
        res = []
        while q:
            cur = q.pop(0)
            res.append(cur)
            if cur in adj:
                for i in adj[cur]:
                    inDegree[i] -= 1
                    if inDegree[i] == 0:
                        q.append(i)
        return res if len(res)==numCourses else []

s = Solution()
a = s.findOrder(2,[])