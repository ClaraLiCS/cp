#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef unsigned long long ull;
typedef pair<int,int> pii;

#define print(s) cout << s << endl //imprimir por pantalla una linea
#define sc(s) getline(cin, s); //leer una linea entera
#define REP(i, a, b) for(int i = a; i < b; ++i) //loop normal
#define REPV(i, a, b) for(int i = b-1; i >= a, i--) //loop descendiente
#define it(i, estructura) for(const auto& i : estructura) //para iterar sobre maps, sets, etc. (solo lectura)
#define find(x, set) (set.find(x) != set.end()) //encontrar un elemento en un set
#define printElem(l) it(i,l){cout << i << " ";} cout << endl; //imprimir los elementos de una lista (array, vector o cualquier cosa que se pueda iterar)
#define ArraySort(a) sort(a, a + (sizeof(a)/sizeof(a[0]))) //ordenar un array
#define VectorSort(v) sort(v.begin(), v.end()) //ordenar un vector

int main() {
    
    
    return 0;
}



//Funciones que podrían ser útiles

int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

int lcm(int a, int b) {
    return (a / gcd(a, b)) * b; // Evita desbordamiento multiplicando después de la división
}

int binarySearch(vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2; // Evita desbordamiento

        if (arr[mid] == target)
            return mid; // Elemento encontrado, devolver índice
        else if (arr[mid] < target)
            left = mid + 1; // Buscar en la mitad derecha
        else
            right = mid - 1; // Buscar en la mitad izquierda
    }

    return -1; // Elemento no encontrado
}

void quickSort(vector<int>& arr, int left, int right) {
    if (left >= right) return;
    int pivot = arr[left + (right - left) / 2], i = left, j = right;
    while (i <= j) {
        while (arr[i] < pivot) i++;
        while (arr[j] > pivot) j--;
        if (i <= j) swap(arr[i++], arr[j--]);
    }
    quickSort(arr, left, j);
    quickSort(arr, i, right);
}

void dfs(int node, vector<vector<int>>& adj, vector<bool>& visited) {
    // Marcar nodo como visitado
    visited[node] = true;
    //doSomething(node);
    
    // Visitar recursivamente los nodos adyacentes a este
    for (int neighbor : adj[node]) {
        if (!visited[neighbor]) {
            dfs(neighbor, adj, visited);
        }
    }
}

void bfs(int nodo_ini, const vector<vector<int>> grafo) {
    // Creamos una cola para almacenar los nodos a visitar
    queue<int> q;

    // Vector para guardar los nodos visitados, se inicializa con false (no hay ningún nodo visitado)
    vector<bool> visitado(grafo.size(), false);

    // Marcamos el nodo inicial como visitado y lo añadimos a la cola
    visitado[nodo_ini] = true;
    q.push(nodo_ini);

    // Mientras la cola no esté vacía, seguimos explorando el grafo
    while (!q.empty()) {
    // Extraemos el siguiente nodo de la cola
        int act_node = q.front();
        q.pop();

        // Recorremos todos los nodos vecinos al nodo actual
        for (int vecinos : grafo[nodo_ini]) {
            // Si el nodo adyacente no ha sido visitado, lo marcamos y lo añadimos a la cola
            if (!visitado[vecinos]) {
                visitado[vecinos] = true;
                q.push(vecinos);
            }
        }
    }
}

void dijkstra(int source, const vector<vector<pii>>& adj, vector<int>& dist) {
    int V = adj.size(); // vértices en el grafo

    // Cola de prioridad
    priority_queue<pii, vector<pii>, greater<pii>> pq;

    // Inicializa distances all vertices as infinite
    dist.assign(V, std::numeric_limits<int>::max());
    dist[source] = 0; // Distancia al nodo origen es 0
    pq.push({0, source});

    // Procesa la cola de propiedad
    while (!pq.empty()) {
        int u = pq.top().second; // Get the vertex with the smallest distance
        int d = pq.top().first;  // Get the distance of the vertex
        pq.pop(); // Remove the vertex from the priority queue

        // Skip processing if the distance is not optimal (stale entry)
        if (d > dist[u]) continue;

        // Traverse all neighbors of the vertex u
        for (const auto& edge : adj[u]) {
            int v = edge.first;    // Neighbor vertex
            int weight = edge.second; // Edge weight

            // Relaxation step: update the distance if a shorter path is found
            if (dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                pq.push({dist[v], v}); 
            }
        }
    }
}

struct SegmentTree {
    vector<int> tree;
    int n;

    SegmentTree(vector<int>& arr) {
        n = arr.size();
        tree.resize(4 * n);
        build(arr, 0, 0, n - 1);
    }

    void build(vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
        } else {
            int mid = (start + end) / 2;
            build(arr, 2 * node + 1, start, mid);
            build(arr, 2 * node + 2, mid + 1, end);
            tree[node] = tree[2 * node + 1] + tree[2 * node + 2]; // Suma
        }
    }

    int maxSum(int node, int start, int end, int L, int R) {
        if (R < start || end < L) return 0;  // Fuera del rango
        if (L <= start && end <= R) return tree[node]; // Dentro del rango
        int mid = (start + end) / 2;
        return maxSum(2 * node + 1, start, mid, L, R) + 
               maxSum(2 * node + 2, mid + 1, end, L, R);
    }

    void update(int node, int start, int end, int idx, int value) {
        if (start == end) {
            tree[node] = value;
        } else {
            int mid = (start + end) / 2;
            if (idx <= mid)
                update(2 * node + 1, start, mid, idx, value);
            else
                update(2 * node + 2, mid + 1, end, idx, value);
            tree[node] = tree[2 * node + 1] + tree[2 * node + 2];
        }
    }
};


//Longest Increasing Subsequence
int LIS(vector<int>& arr) {
    vector<int> dp;
    for (int num : arr) {
        auto it = lower_bound(dp.begin(), dp.end(), num);
        if (it == dp.end()) dp.push_back(num);
        else *it = num;
    }
    return dp.size();
}

//Criba de Eratóstenes
vector<bool> sieve(int n) {
    vector<bool> is_prime(n + 1, true);
    is_prime[0] = is_prime[1] = false;
    for (int i = 2; i * i <= n; i++) {
        if (is_prime[i]) {
            for (int j = i * i; j <= n; j += i)
                is_prime[j] = false;
        }
    }
    return is_prime;
}

//Suma máxima en una ventana de tamaño k
int maxSlidingWindow(vector<int>& arr, int k) {
    int maxSum = 0, windowSum = 0;
    for (int i = 0; i < k; i++) windowSum += arr[i];
    maxSum = windowSum;
    for (int i = k; i < arr.size(); i++) {
        windowSum += arr[i] - arr[i - k];
        maxSum = max(maxSum, windowSum);
    }
    return maxSum;
}

//Distancia entre dos puntos en un plano
double distance(double x1, double y1, double x2, double y2) {
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

//Área de una figura poligonal simple (O(n))
double polygonArea(vector<pair<double, double>>& points) {
    int n = points.size();
    double area = 0;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        area += points[i].first * points[j].second;
        area -= points[j].first * points[i].second;
    }
    return abs(area) / 2.0;
}

//Factorial: Calcula el factorial de un número n (n!)
long long factorial(int n) {
    return (n == 0) ? 1 : n * factorial(n - 1);
}

//Combinaciones: Número de formas de elegir k elementos de un conjunto de n (nCk)
long long combination(int n, int k) {
    return factorial(n) / (factorial(k) * factorial(n - k));
}

//Permutaciones: Número de formas de ordenar k elementos de un conjunto de n (nPk)
long long permutation(int n, int k) {
    return factorial(n) / factorial(n - k);
}

//Coeficiente Binomial: Calcula combinaciones usando programación dinámica para eficiencia
long long binomialCoefficient(int n, int k) {
    vector<vector<long long>> C(n + 1, vector<long long>(k + 1, 0));
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= min(i, k); j++) {
            if (j == 0 || j == i) C[i][j] = 1;
            else C[i][j] = C[i - 1][j - 1] + C[i - 1][j];
        }
    }
    return C[n][k];
}
