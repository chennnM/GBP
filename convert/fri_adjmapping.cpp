#include <iostream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <queue>
#include <map>
#include <unordered_map>
#include <math.h>
#include <cmath>
#include <limits.h>
#include <unistd.h>
#include <math.h>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <thread>
#include <string>
#include <unistd.h>
#include <sys/time.h>

using namespace std;

int main(int argc, char* argv[]){
    int n = 65608366;
    int m = 1806067135;
    
    ifstream infile("com-friendster.ungraph.txt");
    assert(infile.is_open());

    vector<pair<int, int> > edgelist(m);
    int idx = 0;
    int maxnodeid = 0;
    int fromNode,toNode;
    while(infile >> fromNode >> toNode){
        edgelist[idx++] = pair<int, int>(fromNode, toNode);
        maxnodeid = max(maxnodeid,fromNode);
        maxnodeid = max(maxnodeid,toNode);
    }
    infile.close();
    cout<<"m="<<idx<<endl;
    idx = 0;
    vector<int> flag(maxnodeid,-1);
    for( int i = 0 ; i < m ; i++ ){
        int u = edgelist[i].first;
        int v = edgelist[i].second;
        if(flag[u] == -1){
            flag[u] = idx++;
        }
        if(flag[v] == -1){
            flag[v] = idx++;
        }
        edgelist[i].first = flag[u];
        edgelist[i].second = flag[v];
    }
    cout<<"n="<<idx+1<<endl;

    ofstream outgraph("friendster.txt");
    outgraph<<n<<endl;
    for( int i = 0 ; i < m ; i++ ){
        outgraph<<edgelist[i].first<<" "<<edgelist[i].second<<endl;
    }
    outgraph.close();

    return 0;
}