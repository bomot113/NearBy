NearBy - Quora Programming challenge Solution
======
Topics on Quora have location data optionally associated with them, allowing the Quora Nearby feature for our iPhone app. Each question on Quora can have one or more topics associated with it. This feature allows us to display topics and questions near different locations, be it the user's current location, or a specified location on the map.
Your task will be to write a program that will be able to find topics or questions that are near to a given input location, up to a specified limit.
Input format (read from STDIN):
The first line of input will be 3 integers: number of topics T, number of questions Q, and number of queries N.
There will be T lines following that, each with a topic id integer, and two doubles representing that topic's location (you can consider the points to be located on a XY plane, location of a entity is in form of its x and y coordinate in the plane).
There will be Q lines following that, each with a question id integer and number of topics the question is associated with, Qn, followed by Qn number of topic ids associated with that question. These integers are guaranteed to be well-formed existing topics.  There may be 0 topics associated with a question, and in such cases, the question should never appear in the query results.
The next N lines will consist of a char, an integer and two doubles each, representing the type of results required ("t" for topic or "q" for question), the number of results required, and the location to be used as the query.

Sample Input:
3 6 2
0 0.0 0.0
1 1.0 1.0
2 2.0 2.0
0 1 0
1 2 0 1
2 3 0 1 2
3 0
4 0
5 2 1 2
t 2 0.0 0.0
q 5 100.0 100.0


Output format (write to STDOUT):
For each query line given in the input, you are to output the ids of the nearest entities in ascending order of distance from the query location, up to the specified number of results.  When there is a tie between different entities in the ordering due to equal distances (threshold of 0.001 for equality comparisons), the entity with the larger id should be ranked first.

Distance of a question from a point is the minimum of the distance of all topics associated with that question.

Sample Output:
0 1
5 2 1 0
