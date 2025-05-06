#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef unsigned long long ull;
typedef pair<int,int> pii;

#define print(s) cout << s << endl //imprimir por pantalla una linea
#define sc(s) getline(cin, s); //leer una linea entera
#define REP(i, a, b) for(int i = a; i < b; ++i) //loop normal
#define REPV(i, a, b) for(int i = b-1; i >= a; i--) //loop descendiente
#define it(i, estructura) for(const auto& i : estructura) //para iterar sobre maps, sets, etc. (solo lectura)
#define found(x, set) (set.find(x) != set.end()) //encontrar un elemento en un set
#define printElem(l) it(i,l){cout << i << " ";} cout << endl; //imprimir los elementos de una lista (array, vector o cualquier cosa que se pueda iterar)
#define ArraySort(a) sort(a, a + (sizeof(a)/sizeof(a[0]))) //ordenar un array
#define VectorSort(v) sort(v.begin(), v.end()) //ordenar un vector
#define d(x) cout << #x << ": " << x << endl;

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

unordered_map<int, long long> memo1;
// Función para calcular el número de Fibonacci con memoización
long long fibonacci(int n) {
    if (n <= 1) {
        return n;  // Caso base
    }
    if (found(n, memo1)) {
        return memo1[n];  // Devolvemos el resultado almacenado si ya fue calculado
    }
    long long result = fibonacci(n - 1) + fibonacci(n - 2);  // Calculamos el resultado recursivamente
    memo1[n] = result;  // Almacenamos el resultado en el mapa
    return result;
}

vector<std::vector<int>> memo2;
// Función para calcular el número de rutas en una cuadrícula
int numberOfPaths(int m, int n) {
    if (m == 0 || n == 0) {
        return 1;  // Caso base: solo hay una ruta cuando estamos en el borde de la cuadrícula
    }
    if (memo2[m][n] != 0) {
        return memo2[m][n];  // Devolvemos el resultado almacenado si ya fue calculado
    }
    // Calculamos el número de rutas sumando las rutas desde arriba y desde la izquierda
    memo2[m][n] = numberOfPaths(m - 1, n) + numberOfPaths(m, n - 1);
    return memo2[m][n];
}

// 1. Disjoint Set Union (Union-Find)
class DSU {
    vector<int> parent, rank;
public:
    DSU(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; i++) parent[i] = i;
    }
    // Encuentra la raiz del conjunto al que pertenece x (con path compression)
    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
    }
    // Une los conjuntos de x e y (union by rank)
    void unite(int x, int y) {
        int xr = find(x);
        int yr = find(y);
        if (xr == yr) return;
        if (rank[xr] < rank[yr])
            parent[xr] = yr;
        else if (rank[xr] > rank[yr])
            parent[yr] = xr;
        else {
            parent[yr] = xr;
            rank[xr]++;
        }
    }
};

// 2. Dynamic Programming: Fibonacci (Tabulation)
int dpFibonacci(int n) {
    vector<int> dp(n+1);
    dp[0] = 0; dp[1] = 1;
    for (int i = 2; i <= n; i++)
        dp[i] = dp[i-1] + dp[i-2];
    return dp[n];
}

// 3. Trees: Inorder Traversal
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
void inorder(TreeNode* root) {
    if (!root) return;
    inorder(root->left);
    cout << root->val << ' ';
    inorder(root->right);
}

// 4. Rangos: Segment Tree (Range Sum Query)
class SegmentTree {
    vector<int> tree;
    int n;
    void build(vector<int>& arr, int v, int tl, int tr) {
        if (tl == tr)
            tree[v] = arr[tl];
        else {
            int tm = (tl + tr) / 2;
            build(arr, v*2, tl, tm);
            build(arr, v*2+1, tm+1, tr);
            tree[v] = tree[v*2] + tree[v*2+1];
        }
    }
    int sum(int v, int tl, int tr, int l, int r) {
        if (l > r) return 0;
        if (l == tl && r == tr) return tree[v];
        int tm = (tl + tr) / 2;
        return sum(v*2, tl, tm, l, min(r, tm))
             + sum(v*2+1, tm+1, tr, max(l, tm+1), r);
    }
public:
    SegmentTree(vector<int>& arr) {
        n = arr.size();
        tree.resize(4*n);
        build(arr, 1, 0, n-1);
    }
    int rangeSum(int l, int r) {
        return sum(1, 0, n-1, l, r);
    }
};

// 5. Matrices: 2D Prefix Sum
vector<vector<int>> computePrefixSum(const vector<vector<int>>& matrix) {
    int m = matrix.size(), n = matrix[0].size();
    vector<vector<int>> prefix(m+1, vector<int>(n+1, 0));
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            prefix[i][j] = matrix[i-1][j-1] + prefix[i-1][j]
                           + prefix[i][j-1] - prefix[i-1][j-1];
        }
    }
    return prefix;
}

// 6. DFS: Connected Components
void dfs(int u, vector<vector<int>>& adj, vector<bool>& visited) {
    visited[u] = true;
    for (int v : adj[u])
        if (!visited[v]) dfs(v, adj, visited);
}

// 7. Matemáticas: Criba de Eratóstenes
vector<bool> sieve(int n) {
    vector<bool> is_prime(n+1, true);
    is_prime[0] = is_prime[1] = false;
    for (int i = 2; i*i <= n; i++) {
        if (is_prime[i]) {
            for (int j = i*i; j <= n; j += i)
                is_prime[j] = false;
        }
    }
    return is_prime;
}

// 8. Geometría: Convex Hull (Graham Scan)
struct Point {
    int x, y;
    bool operator<(const Point& p) const {
        return tie(x, y) < tie(p.x, p.y);
    }
};
int cross(const Point& O, const Point& A, const Point& B) {
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}
vector<Point> convexHull(vector<Point>& P) {
    int n = P.size(), k = 0;
    vector<Point> H(2*n);
    sort(P.begin(), P.end());
    // lower hull
    for (int i = 0; i < n; ++i) {
        while (k >= 2 && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
        H[k++] = P[i];
    }
    // upper hull
    for (int i = n-2, t = k+1; i >= 0; --i) {
        while (k >= t && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
        H[k++] = P[i];
    }
    H.resize(k-1);
    return H;
}

// 9. Priority Queue: Dijkstra
void dijkstra(int src, vector<vector<pair<int, int>>>& adj, vector<int>& dist) {
    int n = adj.size();
    dist.assign(n, INT_MAX);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
    dist[src] = 0;
    pq.emplace(0, src);
    while (!pq.empty()) {
        int d, u;
        tie(d, u) = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto& edge : adj[u]) {
            int v = edge.first, w = edge.second;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.emplace(dist[v], v);
            }
        }
    }
}
