/*
 * vi:ts=4:tw=78:shiftwidth=4:expandtab
 * vim600:fdm=marker
 */

#ifndef P_QUEUE_H
#define P_QUEUE_H

#include <deque>
#include <string>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;

template <class Data>
/// Priority queue
class PQueue {
    protected:
        double threshold;

    public:
		size_t n;
		deque<pair<double,Data> > queue;

        PQueue(size_t csize)
        {
            n = csize;
            threshold = 0;
        }

        void print()
        {
            printf("Show =========\n");
            for (size_t i=0; i < queue.size(); i++) {
				if (i == n) printf("--------------\n");
                printf("key=%d value=%g\n", queue[i].second, queue[i].first);
				if (i >= n && queue[i].first > threshold) {
					printf("PQueue Error\n");
					exit(1);
				}
			}
        }

        void insert(pair<double,Data> item)
        {
            if (queue.size() < n) {
                if (item.first < threshold || queue.size() == 0) {
                    threshold = item.first;
                    queue.push_back(item);
                }
                else {
                    queue.push_front(item);
                }
            }
            else if (item.first > threshold)
            {
                queue.push_front(item);
				// find minimum item
				double min_value = item.first;
				Data min_data = item.second;
				size_t min_i = 0;
				for (size_t i=0; i < n; i++) {
					if (queue[i].first < min_value) {
						min_i = i;
						min_data = queue[i].second;
						min_value = queue[i].first;
					}
				}
				threshold = min_value;
				// swap min_i, n-1
				queue[min_i].first = queue[n-1].first;
				queue[min_i].second = queue[n-1].second;
				queue[n-1].first = min_value;
				queue[n-1].second = min_data;
            }
        }

        size_t size()
        {
            return queue.size();
        }
};
#endif // P_QUEUE_H
