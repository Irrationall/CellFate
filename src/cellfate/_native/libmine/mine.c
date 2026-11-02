/*
 * Libmine core library.
 *
 * This code is written by Davide Albanese <davide.albanese@gmail.com>
 * and Michele Filosi <filosi@fbk.eu>.
 *
 * Copyright (C) 2012-2016 Davide Albanese, Copyright (C) 2012 Michele
 * Filosi, Copyright (C) 2012 Fondazione Bruno Kessler.
 *
 * References:
 *   * Original MINE paper: DOI: 10.1126/science.1205438;
 *   * Minepy paper: DOI: 10.1093/bioinformatics/bts707D;
 *   * MIC_e and TIC: DOI: arXiv:1505.02213 and DOI: arXiv:1505.02214;
 *   * GMIC: DOI: arXiv:1308.5712.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "mine.h"

char *libmine_version = LIBMINE_VERSION;

#define MAX(a, b) (((a) > (b) ? (a) : (b)))
#define MIN(a, b) (((a) < (b) ? (a) : (b)))

#define TRUE 1
#define FALSE 0


void quicksort(double *a, int *idx, int l, int u)
{
  int i, m, idx_temp;
  double a_temp;

  if (l >= u)
    return;

  m = l;
  for (i=l+1; i<=u; i++)
    {
      if (a[i] < a[l])
        {
          ++m;

          idx_temp = idx[m];
          idx[m] = idx[i];
          idx[i] = idx_temp;

          a_temp = a[m];
          a[m] = a[i];
          a[i] = a_temp;
        }
    }

  idx_temp = idx[l];
  idx[l] = idx[m];
  idx[m] = idx_temp;

  a_temp = a[l];
  a[l] = a[m];
  a[m] = a_temp;

  quicksort(a, idx, l, m-1);
  quicksort(a, idx, m+1, u);
}


int *argsort(double *a, int n)
{
  double *a_cpy;
  int i, *idx;

  a_cpy = (double *) malloc(n * sizeof(double));
  if (a_cpy == NULL)
    return NULL;

  idx = (int *) malloc(n * sizeof(int));
  if (idx == NULL)
    {
      free(a_cpy);
      return NULL;
    }

  /* fill a_cpy */
  memcpy(a_cpy, a, n * sizeof(double));

  /* fill idx */
  for (i=0; i<n; i++)
    idx[i] = i;

  quicksort(a_cpy, idx, 0, n-1);
  free(a_cpy);

  return idx;
}


/*
 * Returns the entropy induced by the points on the partition Q.
 * See section 3.2.1, page 10, SOM.
 *
 * Parameters
 *   cumhist : cumulative histogram matrix along P_map
 *   cumhist_log : log(cumhist)
 *   q : number of rows of cumhist (number of partitions in Q_map)
 *   p : number of cols of cumhist (number of partitions in P_map)
 *   n : total number of points
 */
double hq(int **cumhist, double **cumhist_log, int q, int p, int n)
{
  int i;
  double total, total_log, prob, prob_log, H = 0.0;

  total = (double) n;
  total_log = log(total);

  for (i=0; i<q; i++)
    {
      prob = (double) cumhist[i][p-1] / total;
      if (prob != 0)
        {
          prob_log = cumhist_log[i][p-1] - total_log;
          H -= prob * prob_log;
       }
    }

  return H;
}

/*
 * Returns the entropy induced by the points on the partition
 * <c_0, c_s, c_t>. See line 5 of Algorithm 2, SOM.
 *
 * Parameters
 *   c : c_1, ..., c_p
 *   c_log : log(c)
 *   s : s in c_s
 *   t : t in c_t
 */
double hp3(int *c, double *c_log, int s, int t)
{
  int sum;
  double total, total_log, prob, prob_log, H = 0.0;

  if (s == t)
    return 0.0;

  total = (double) c[t-1];
  total_log = log(total);

  prob = (double) c[s-1] / total;
  if (prob != 0)
    {
      prob_log = c_log[s-1] - total_log;
      H -= prob * prob_log;
    }

  sum = c[t-1] - c[s-1];
  prob = (double) sum / total;
  if (sum != 0)
    {
      prob_log = log((double) sum) - total_log;
      H -= prob * prob_log;
    }

  return H;
}


/*
 * Returns the entropy induced by the points on the partition
 * <c_0, c_s, c_t>, Q. See line 5 of Algorithm 2 in SOM.
 *
 * Parameters
 *   cumhist : cumulative histogram matrix along P_map
 *   cumhist_log : log(cumhist)
 *   c : c_1, ..., c_p
 *   q : number of rows of cumhist (number of partitions in Q_map)
 *   p : number of cols of cumhist (number of partitions in P_map)
 *   s : s in c_s
 *   t : t in c_t
 */
double hp3q(int **cumhist, double **cumhist_log, int *c, int q, int p, int s, int t)

{
  int i, sum;
  double total, total_log, prob, prob_log, H = 0.0;

  total = (double) c[t-1];
  total_log = log(total);

  for (i=0; i<q; i++)
    {
      prob = (double) cumhist[i][s-1] / total;
      if (prob != 0)
        {
          prob_log = cumhist_log[i][s-1] - total_log;
          H -= prob * prob_log;
        }

      sum = cumhist[i][t-1] - cumhist[i][s-1];
      prob = (double) sum / total;
      if (prob != 0)
        {
          prob_log = log((double) sum) - total_log;
          H -= prob * prob_log;
        }
    }

  return H;
}

/*
 * Returns the entropy induced by the points on the partition <c_s, c_t>
 * and Q. See line 13 of Algorithm 2, SOM.
 *
 * Parameters
 *   cumhist : cumulative histogram matrix along P_map
 *   c : c_1, ..., c_p
 *   q : number of rows of cumhist (number of partitions
 *       in Q_map)
 *   p : number of cols of cumhist (number of partitions
 *       in P_map)
 *   s : s in c_s
 *   t : t in c_t
 */
double hp2q(int **cumhist, int *c, int q, int p, int s, int t)
{
  int i, sum;
  double total, total_log, prob, prob_log, H = 0.0 ;

  if (s == t)
    return 0.0;

  total = (double) (c[t-1] - c[s-1]);
  total_log = log(total);

  for (i=0; i<q; i++)
    {
      sum = cumhist[i][t-1] - cumhist[i][s-1];
      prob = (double) sum / total;
      if (prob != 0)
        {
          prob_log = log((double) sum) - total_log;
          H -= prob * prob_log;
        }
    }

  return H;
}

/*
 * Returns the map Q: D -> {0, ...,q-1}.
 * See Algorithm 3 in SOM.
 *
 * Parameters
 *   dy (IN): y-data sorted in increasing order
 *   n (IN): number of elements of dy
 *   y (IN): an integer greater than 1
 *   Q_map (OUT) : the map Q. Q_map must be a preallocated vector of
 *                 size n
 *   q (OUT) : number of partitions in Q_map. q can be < y
 *
 * Returns
 *   0
 */
int EquipartitionYAxis(double *dy, int n, int y, int *Q_map, int *q)
{
  int i, j, s, h, curr;
  double temp1, temp2;

  double rowsize = (double) n / (double) y;

  i = 0;
  h = 0;
  curr = 0;

  while (i < n)
    {
      s = 1;
      for (j=i+1; j<n; j++)
        {
          if (dy[i] == dy[j])
            ++s;
          else
            break;
        }

      temp1 = fabs((double) h + (double) s - rowsize);
      temp2 = fabs((double) h - rowsize);
      if ((h != 0) && (temp1 >= temp2))
        {
          ++curr;
          h = 0;
          temp1 = (double) n - (double) i;
          temp2 = (double) y - (double) curr;
          rowsize = temp1 / temp2;
        }

      for (j=0; j<s; j++)
         Q_map[i+j] = curr;

      i += s;
      h += s;
    }

  *q = curr + 1;

  return 0;
}

/*
 * Returns the map P: D -> {0, ...,p-1}.
 *
 * Parameters
 *   dx (IN) : x-data sorted in increasing order
 *   n (IN) : number of elements of dx
 *   Q_map (IN) : the map Q computed by EquipartitionYAxis sorted in
 *                increasing order by dx-values
 *   P_map (OUT) : the map P. P_map must be a preallocated vector
 *                 of size n
 *   p (OUT) : number of partitions in P_map
 *
 * Returns
 *   0 on success, 1 if an error occurs
 */
int GetClumpsPartition(double *dx, int n, int *Q_map, int *P_map, int *p)
{
  int i, j, flag, c, s;
  int *Q_tilde;

  i = 0;
  c = -1;

  Q_tilde = (int *) malloc (n * sizeof(int));
  if (Q_tilde == NULL)
    return 1;

  memcpy(Q_tilde, Q_map, n*sizeof(int));

  while (i < n)
    {
      s = 1;
      flag = FALSE;
      for (j=i+1; j<n; j++)
        {
          if (dx[i] == dx[j])
            {
              if (Q_tilde[i] != Q_tilde[j])
                flag = TRUE;
              ++s;
            }
          else
            break;
        }

      if ((s > 1) && (flag == TRUE))
        {
          for (j=0; j<s; j++)
            Q_tilde[i+j] = c;
          --c;
        }

      i += s;
    }

  i = 0;
  P_map[0] = 0;
  for (j=1; j<n; j++)
    {
      if (Q_tilde[j] != Q_tilde[j-1])
        ++i;
      P_map[j] = i;
    }

  *p = i + 1;
  free(Q_tilde);

  return 0;
}

/*
 * Returns the map P: D -> {0, ...,p-1}.
 *
 * Parameters
 *   dx (IN) : x-data sorted in increasing order
 *   n (IN) : number of elements of dx
 *   k_hat (IN) : maximum number of clumps
 *   Q_map (IN) : the map Q computed by EquipartitionYAxis sorted in
 *                increasing order by dx-values
 *   P_map (OUT) : the map P. P_map must be a preallocated vector
 *                 of size n
 *   p (OUT) : number of partitions in P_map
 *
 * Returns
 *   0 on success, 1 if an error occurs
 */
int GetSuperclumpsPartition(double *dx, int n, int k_hat, int *Q_map,
                            int *P_map, int *p)
{
  int i, ret;
  double *dp;

  /* clumps */
  ret = GetClumpsPartition(dx, n, Q_map, P_map, p);
  if (ret)
    return 1;

  /* superclumps */
  if (*p > k_hat)
    {
      dp = (double *) malloc (n * sizeof(double));
      if (dp == NULL)
        return 1;

      for (i=0; i<n; i++)
        dp[i] = (double) P_map[i];

      EquipartitionYAxis(dp, n, k_hat, P_map, p);

      free(dp);
    }

  return 0;
}


/* Returns (c_1, ..., c_k) */
int *compute_c(int *P_map, int p, int n)
{
  int i;
  int *c;


  c = (int *) malloc (p * sizeof(int));
  if (c == NULL)
    return NULL;

  for (i=0; i<p; i++)
    c[i] = 0;

  for (i=0; i<n; i++)
    c[P_map[i]]++;

  for (i=1; i<p; i++)
    c[i] += c[i-1];

  return c;
}


double *compute_c_log(int *c, int p)
{
  int i;
  double *c_log;

  c_log = (double *) malloc (p * sizeof(double));
  if (c_log == NULL)
    return NULL;

  for (i=0; i<p; i++)
    if (c[i] != 0)
      c_log[i] = log((double) c[i]);
    else
      c_log[i] = 0;

  return c_log;
}


/* Returns the cumulative histogram matrix along P_map */
int **compute_cumhist(int *Q_map, int q, int *P_map, int p, int n)
{
  int i, j;
  int **cumhist;


  cumhist = (int **) malloc (q * sizeof(int *));
  if (cumhist == NULL)
    return NULL;

  for (i=0; i<q; i++)
    {
      cumhist[i] = (int *) malloc (p * sizeof(int));
      if (cumhist[i] == NULL)
        {
          for (j=0; j<i; j++)
            free(cumhist[j]);

          free(cumhist);
          return NULL;
        }

      for (j=0; j<p; j++)
        cumhist[i][j] = 0;
    }

  for (i=0; i<n; i++)
    cumhist[Q_map[i]][P_map[i]]++;

  for (i=0; i<q; i++)
    for (j=1; j<p; j++)
      cumhist[i][j] += cumhist[i][j-1];

  return cumhist;
}


double ** compute_cumhist_log(int **cumhist, int q, int p)
{
  int i, j;
  double **cumhist_log;

  cumhist_log = (double **) malloc (q * sizeof(double *));
  if (cumhist_log == NULL)
    return NULL;

  for (i=0; i<q; i++)
    {
      cumhist_log[i] = (double *) malloc (p * sizeof(double));
      if (cumhist_log[i] == NULL)
        {
          for (j=0; j<i; j++)
            free(cumhist_log[j]);

          free(cumhist_log);
          return NULL;
        }

      for (j=0; j<p; j++)
        if (cumhist[i][j] != 0)
          cumhist_log[i][j] = log((double) cumhist[i][j]);
        else
          cumhist_log[i][j] = 0;
    }

    return cumhist_log;
}


/* Initializes the I matrix */
double **init_I(int p, int x)
{
  int i, j;
  double **I;


  I = (double **) malloc ((p+1) * sizeof(double *));
  if (I == NULL)
    return NULL;

  for (i=0; i<=p; i++)
    {
      I[i] = (double *) malloc ((x+1) * sizeof(double));
      if (I[i] == NULL)
        {
          for (j=0; j<i; j++)
            free(I[j]);
          free(I);
          return NULL;
        }

      for (j=0; j<=x; j++)
        I[i][j] = 0.0;
    }

  return I;
}


/* Computes the HP2Q matrix */
double **compute_HP2Q(int **cumhist, int*c, int q, int p)
{
  int i, j, s, t;
  double **HP2Q;


  HP2Q = (double **) malloc ((p+1) * sizeof(double *));
  if (HP2Q == NULL)
    return NULL;

  for (i=0; i<=p; i++)
    {
      HP2Q[i] = (double *) malloc ((p+1) * sizeof(double));
      if (HP2Q[i] == NULL)
        {
          for (j=0; j<i; j++)
            free(HP2Q[j]);
          free(HP2Q);
          return NULL;
        }
    }

  for (t=3; t<=p; t++)
    for (s=2; s<=t; s++)
      HP2Q[s][t] = hp2q(cumhist, c, q, p, s, t);

  return HP2Q;
}


/*
 * Returns the normalized MI scores.
 *
 * Parameters
 *   dx (IN) : x-data sorted in increasing order by dx-values
 *   dy (IN) : y-data sorted in increasing order by dx-values
 *   n (IN) : number of elements in dx and dy
 *   Q_map (IN) : the map Q computed by EquipartitionYAxis() sorted in
 *                increasing order by dx-values
 *   q (IN) : number of partitions in Q_map
 *   P_map (IN) : the map P computed by GetSuperclumpsPartition() sorted
 *                in increasing order by Dx-values
 *   p (IN) : number of partitions in P_map
 *   x (IN) : maximum grid size on dx-values
 *   score (OUT) : mutual information scores. score must be a
 *                 preallocated array of dimension x-1
 * Returns
 *   0 on success, 1 if an error occurs
 */
int OptimizeXAxis(double *dx, double *dy, int n, int *Q_map, int q,
      int *P_map, int p, int x, double *score)
{
  int i, s, t, l;
  int *c;
  int **cumhist;
  double **I, **HP2Q;
  double F, F_max, HQ, ct, cs;
  double **cumhist_log, *c_log;

  /* return score=0 if p=1 */
  if (p == 1)
    {
      for (i=0; i<x-1; i++)
        score[i] = 0.0;
      return 0;
    }

  /* compute c */
  c = compute_c(P_map, p, n);
  if (c == NULL)
    goto error_c;

  /* precompute log(c) (log(c)=0 when c=0) */
  c_log = compute_c_log(c, p);
  if (c_log == NULL)
    goto error_c_log;

  /* compute the cumulative histogram matrix along P_map */
  cumhist = compute_cumhist(Q_map, q, P_map, p, n);
  if (cumhist == NULL)
    goto error_cumhist;

  /* precompute log(cumhist) (log(cumhist)=0 when cumhist=0) */
  cumhist_log = compute_cumhist_log(cumhist, q, p);
  if (cumhist == NULL)
    goto error_cumhist_log;
  
  /* I matrix initialization */
  I = init_I(p, x);
  if (I == NULL)
    goto error_I;

  /* Precomputes the HP2Q matrix */
  HP2Q = compute_HP2Q(cumhist, c, q, p);
  if (HP2Q == NULL)
    goto error_HP2Q;

  /* compute H(Q) */
  HQ = hq(cumhist, cumhist_log, q, p, n);

  /* Find the optimal partitions of size 2, Algorithm 2 in SOM, lines 3-8 */
  for (t=2; t<=p; t++)
    {
      F_max = -DBL_MAX;
      for (s=1; s<=t; s++)
        {
          F = hp3(c, c_log, s, t) - hp3q(cumhist, cumhist_log, c, q, p, s, t);
          if (F > F_max)
            {
              I[t][2] = HQ + F;
              F_max = F;
            }
        }
    }

  /*
   * Inductively build the rest of the table of optimal partitions,
   * Algorithm 2 in SOM, lines 10-17
   */
  for (l=3; l<=x; l++)
    {
      for (t=l; t<=p; t++)
        {
          ct = (double) c[t-1];
          F_max = -DBL_MAX;
          for (s=l-1; s<=t; s++)
            {
              cs = (double) c[s-1];
              F = ((cs/ct) * (I[s][l-1]-HQ)) - (((ct-cs)/ct) * HP2Q[s][t]);

              if (F > F_max)
                {
                  I[t][l] = HQ + F;
                  F_max = F;
                }
            }
        }
    }

  /* Algorithm 2 in SOM, line 19 */
  for (i=p+1; i<=x; i++)
    I[p][i] = I[p][p];

  /* score */
  for (i=2; i<=x; i++)
    score[i-2] = I[p][i] / MIN(log(i), log(q));

  /* start frees */
  for (i=0; i<=p; i++)
    free(HP2Q[i]);
  free(HP2Q);

  for (i=0; i<=p; i++)
    free(I[i]);
  free(I);

  for (i=0; i<q; i++)
    free(cumhist_log[i]);
  free(cumhist_log);

  for (i=0; i<q; i++)
    free(cumhist[i]);
  free(cumhist);

  free(c_log);
  free (c);
  /* end frees */

  return 0;

  /* gotos */
  error_HP2Q:
    for (i=0; i<=p; i++)
      free(I[i]);
    free(I);
  error_I:
    for (i=0; i<q; i++)
      free(cumhist_log[i]);
    free(cumhist_log);
  error_cumhist_log: 
    for (i=0; i<q; i++)
      free(cumhist[i]);
    free(cumhist);
  error_cumhist:
    free(c_log);
  error_c_log:
    free(c);
  error_c:
    return 1;
}


/*
 * Returns an initialized mine_score structure. Returns NULL if an error
 * occurs.
 */
mine_score *init_score(mine_problem *prob, mine_parameter *param)
{
  int i, j;
  double B;
  mine_score *score;

  if ((param->alpha > 0.0) && (param->alpha <= 1.0))
    B = MAX(pow(prob->n, param->alpha), 4);
  else if (param->alpha >= 4)
    B = MIN(param->alpha, prob->n);
  else
    goto error_score;

  score = (mine_score *) malloc (sizeof(mine_score));
  if (score == NULL)
    goto error_score;

  score->n = MAX((int) floor(B/2.0), 2) - 1;
  score->m = (int *) malloc(score->n * sizeof(int));
  if (score->m == NULL)
    goto error_score_m;

  for (i=0; i<score->n; i++)
    score->m[i] = (int) floor((double) B / (double) (i+2)) - 1;

  score->M = (double **) malloc (score->n * sizeof(double *));
  if (score->M == NULL)
    goto error_score_M;

  for (i=0; i<score->n; i++)
    {
      score->M[i] = (double *) malloc ((score->m[i]) * sizeof(double));
      if (score->M[i] == NULL)
        {
          for (j=0; j<i; j++)
            free(score->M[j]);
          goto error_score_M_i;
        }
    }

  return score;

  error_score_M_i:
    free(score->M);
  error_score_M:
    free(score->m);
  error_score_m:
    free(score);
  error_score:
    return NULL;
}


/* See mine.h */
mine_score *mine_compute_score(mine_problem *prob, mine_parameter *param)
{
  int i, j, k, p, q, ret;
  double *xx, *yy, *xy, *yx, *M_temp;
  int *ix, *iy;
  int *Q_map_temp, *Q_map, *P_map;
  mine_score *score;

  score = init_score(prob, param);
  if (score == NULL)
    goto error_score;

  xx = (double *) malloc (prob->n * sizeof(double));
  if (xx == NULL)
    goto error_xx;

  yy = (double *) malloc (prob->n * sizeof(double));
  if (yy == NULL)
    goto error_yy;

  xy = (double *) malloc (prob->n * sizeof(double));
  if (xy == NULL)
    goto error_xy;

  yx = (double *) malloc (prob->n * sizeof(double));
  if (yx == NULL)
    goto error_yx;

  Q_map_temp = (int *) malloc (prob->n * sizeof(int));
  if (Q_map_temp == NULL)
    goto error_Q_temp;

  Q_map = (int *) malloc (prob->n * sizeof(int));
  if (Q_map == NULL)
    goto error_Q;

  P_map = (int *) malloc (prob->n * sizeof(int));
  if (P_map == NULL)
    goto error_P;

  ix = argsort(prob->x, prob->n);
  if (ix == NULL)
    goto error_ix;

  iy = argsort(prob->y, prob->n);
  if (iy == NULL)
    goto error_iy;

  M_temp = (double *)malloc ((score->m[0]) * sizeof(double));
  if (M_temp == NULL)
    goto error_M_temp;

  /* build xx, yy, xy, yx */
  for (i=0; i<prob->n; i++)
    {
      xx[i] = prob->x[ix[i]];
      yy[i] = prob->y[iy[i]];
      xy[i] = prob->x[iy[i]];
      yx[i] = prob->y[ix[i]];
    }

  /* x vs. y */
  for (i=0; i<score->n; i++)
    {
      k = MAX((int) (param->c * (score->m[i]+1)), 1);

      ret = EquipartitionYAxis(yy, prob->n, i+2, Q_map, &q);
      if (ret)
        goto error_0;

      /* sort Q by x */
      for (j=0; j<prob->n; j++)
        Q_map_temp[iy[j]] = Q_map[j];
      for (j=0; j<prob->n; j++)
        Q_map[j] = Q_map_temp[ix[j]];

      ret = GetSuperclumpsPartition(xx, prob->n, k, Q_map, P_map, &p);
      if (ret)
        goto error_0;

      if (param->est == EST_MIC_APPROX)
        ret = OptimizeXAxis(xx, yx, prob->n, Q_map, q, P_map, p, score->m[i]+1,
                            score->M[i]);
      else /* EST_MIC_E */
        ret = OptimizeXAxis(xx, yx, prob->n, Q_map, q, P_map, p,
                            MIN(i+2, score->m[i]+1), score->M[i]);
      if (ret)
        goto error_0;
    }

  /* y vs. x */
  for (i=0; i<score->n; i++)
    {
      k = MAX((int) (param->c * (score->m[i]+1)), 1);

      ret = EquipartitionYAxis(xx, prob->n, i+2, Q_map, &q);
      if (ret)
        goto error_0;

      /* sort Q by y */
      for (j=0; j<prob->n; j++)
        Q_map_temp[ix[j]] = Q_map[j];
      for (j=0; j<prob->n; j++)
        Q_map[j] = Q_map_temp[iy[j]];

      ret = GetSuperclumpsPartition(yy, prob->n, k, Q_map, P_map, &p);
      if (ret)
        goto error_0;

      if (param->est == EST_MIC_APPROX)
        ret = OptimizeXAxis(yy, xy, prob->n, Q_map, q, P_map, p, score->m[i]+1,
                            M_temp);
      else /* EST_MIC_E */
        ret = OptimizeXAxis(yy, xy, prob->n, Q_map, q, P_map, p,
                            MIN(i+2, score->m[i]+1), M_temp);

      if (ret)
        goto error_0;

      if (param->est == EST_MIC_APPROX)
        for (j=0; j<score->m[i]; j++)
          score->M[j][i] = MAX(M_temp[j], score->M[j][i]);
      else /* EST_MIC_E */
        for (j=0; j<MIN(i+1, score->m[i]); j++)
          score->M[j][i] = M_temp[j];
    }

  free(M_temp);
  free(iy);
  free(ix);
  free(P_map);
  free(Q_map);
  free(Q_map_temp);
  free(yx);
  free(xy);
  free(yy);
  free(xx);

  return score;

  error_0:
    free(M_temp);
  error_M_temp:
    free(iy);
  error_iy:
    free(ix);
  error_ix:
    free(P_map);
  error_P:
    free(Q_map);
  error_Q:
    free(Q_map_temp);
  error_Q_temp:
    free(yx);
  error_yx:
    free(xy);
  error_xy:
    free(yy);
  error_yy:
    free(xx);
  error_xx:
    for (i=0; i<score->n; i++)
      free(score->M[i]);
    free(score->M);
    free(score->m);
    free(score);
  error_score:
    return NULL;
}


/* See mine.h */
char *mine_check_parameter(mine_parameter *param)
{
  if (((param->alpha <= 0.0) || (param->alpha > 1.0)) && (param->alpha < 4.0))
    return "alpha must be in (0.0, 1.0] or >= 4.0";

  if (param->c <= 0.0)
    return "c must be > 0.0";

  if ((param->est != EST_MIC_APPROX) && (param->est != EST_MIC_E))
    return "unknown estimator";

  return NULL;
}


/* See mine.h */
double mine_mic(mine_score *score)
{
  int i, j;
  double score_max = 0.0;

  for (i=0; i<score->n; i++)
    for (j=0; j<score->m[i]; j++)
      if (score->M[i][j] > score_max)
        score_max = score->M[i][j];

  return score_max;
}


/* See mine.h */
double mine_mas(mine_score *score)
{
  int i, j;
  double score_curr;
  double score_max = 0.0;

  for (i=0; i<score->n; i++)
    for (j=0; j<score->m[i]; j++)
      {
        score_curr = fabs(score->M[i][j] - score->M[j][i]);
        if (score_curr > score_max)
          score_max = score_curr;
      }

  return score_max;
}


/* See mine.h */
double mine_mev(mine_score *score)
{
  int i, j;
  double score_max = 0.0;

  for (i=0; i<score->n; i++)
    for (j=0; j<score->m[i]; j++)
      if (((j==0) || (i==0)) && score->M[i][j] > score_max)
        score_max = score->M[i][j];

  return score_max;
}


/* See mine.h */
double mine_mcn(mine_score *score, double eps)
{
  int i, j;
  double log_xy;
  double score_min = DBL_MAX;
  double delta = 0.0001; /* avoids overestimation of mcn */
  double mic = mine_mic(score);

  for (i=0; i<score->n; i++)
    for (j=0; j<score->m[i]; j++)
      {
        log_xy = log((i+2) * (j+2)) / log(2.0);
        if (((score->M[i][j]+delta) >= ((1.0 - eps) * mic))
             && (log_xy < score_min))
          score_min = log_xy;
      }

  return score_min;
}


/* See mine.h */
double mine_mcn_general(mine_score *score)
{
  int i, j;
  double log_xy;
  double score_min = DBL_MAX;
  double delta = 0.0001; /* avoids overestimation of mcn */
  double mic = mine_mic(score);

  for (i=0; i<score->n; i++)
    for (j=0; j<score->m[i]; j++)
      {
        log_xy = log((i+2) * (j+2)) / log(2.0);
        if (((score->M[i][j]+delta) >= (mic * mic)) && (log_xy < score_min))
          score_min = log_xy;
      }

  return score_min;
}

/* See mine.h */
double mine_tic(mine_score *score, int norm)
{
  int i, j, k=0;
  double tic = 0.0;

  for (i=0; i<score->n; i++)
    for (j=0; j<score->m[i]; j++)
      {
        tic += score->M[i][j];
        k++;
      }

  if (norm)
    tic /= k;

  return tic;
}

/* See mine.h */
double mine_gmic(mine_score *score, double p)
{
  int i, j, k, Z, B;
  mine_score *score_sub, *C_star;
  double gmic;

  /* alloc score_sub */
  score_sub = (mine_score *) malloc (sizeof(mine_score));

  /* alloc C_star */
  C_star = (mine_score *) malloc (sizeof(mine_score));
  C_star->m = (int *) malloc(score->n * sizeof(int));
  C_star->M = (double **) malloc (score->n * sizeof(double *));
  for (i=0; i<score->n; i++)
    C_star->M[i] = (double *) malloc ((score->m[i]) * sizeof(double));

  /* prepare score_sub */
  score_sub->M = score->M;

  /* prepare C_star */
  C_star->n = score->n;
  for (i=0; i<C_star->n; i++)
    C_star->m[i] = score->m[i];

  /* compute C_star */
  for (i=0; i<score->n; i++)
    for (j=0; j<score->m[i]; j++)
      {
        B = (i+2) * (j+2);
        score_sub->n = MAX((int) floor(B/2.0), 2) - 1;
        score_sub->m = (int *) malloc(score_sub->n * sizeof(int));
        for (k=0; k<score_sub->n; k++)
          score_sub->m[k] = (int) floor((double) B / (double) (k+2)) - 1;

        C_star->M[i][j] = mine_mic(score_sub);
        free(score_sub->m);
      }

  /* p=0 -> geometric mean */
  if (p == 0.0)
    {
      Z = 0;
      gmic = 1.0;
      for (i=0; i<C_star->n; i++)
        for (j=0; j<C_star->m[i]; j++)
          {
            gmic *= C_star->M[i][j];
            Z++;
          }
      gmic = pow(gmic, (double) Z);
    }
  /* p!=0 -> generalized mean */
  else
    {
      Z = 0;
      gmic = 0.0;
      for (i=0; i<C_star->n; i++)
        for (j=0; j<C_star->m[i]; j++)
          {
            gmic += pow(C_star->M[i][j], p);
            Z++;
          }
      gmic /= (double) Z;
      gmic = pow(gmic, 1.0/p);
    }

  free(score_sub);
  if (C_star->n != 0)
    {
      free(C_star->m);
      for (i=0; i<C_star->n; i++)
         free(C_star->M[i]);
      free(C_star->M);
    }
  free(C_star);

  return gmic;
}

/* See mine.h */
void mine_free_score(mine_score **score)
{
  int i;
  mine_score *score_ptr = *score;

  if (score_ptr != NULL)
    {
      if (score_ptr->n != 0)
        {
          free(score_ptr->m);
          for (i=0; i<score_ptr->n; i++)
            free(score_ptr->M[i]);
          free(score_ptr->M);
        }

      free(score_ptr);
      score_ptr = NULL;
    }
}


mine_pstats *mine_compute_pstats(mine_matrix *X, mine_parameter *param)
{
  int i, j, k;
  mine_problem prob;
  mine_score *score;
  mine_pstats *stats;

  /* Allocate memory for stats */
  stats = (mine_pstats *) malloc(sizeof(mine_pstats));
  stats->n = (X->n * (X->n-1)) / 2;
  stats->mic = (double *) malloc(stats->n * sizeof(double));
  stats->tic = (double *) malloc(stats->n * sizeof(double));

  k = 0;
  prob.n = X->m;
  for (i=0; i<X->n-1; i++)
    {
      prob.x = &X->data[i*X->m];
      for (j=i+1; j<X->n; j++)
        {
          prob.y = &X->data[j*X->m];
          score = mine_compute_score(&prob, param);
          stats->mic[k] = mine_mic(score);
          stats->tic[k] = mine_tic(score, TRUE);
          mine_free_score(&score);
          k++;
        }
    }

  return stats;
}

typedef struct mine_mic_workspace
{
  int n;
  mine_score *score;
  double *M_temp, *dp;
  int *Q_map, *P_map, *Q_tilde;
  int *c, *cumhist_data;
  double *log_int, *ratio, *plogp, *I_data, *HP2Q_data;
  int **cumhist;
  double **I, **HP2Q;
} mine_mic_workspace;


static mine_mic_workspace *init_mic_workspace(int n, mine_parameter *param)
{
  int i;
  mine_problem prob;
  mine_mic_workspace *ws;

  ws = (mine_mic_workspace *) calloc(1, sizeof(mine_mic_workspace));
  if (ws == NULL)
    return NULL;
  ws->n = n;

  prob.n = n;
  ws->score = init_score(&prob, param);
  ws->M_temp = ws->score == NULL ? NULL :
    (double *) malloc(ws->score->m[0] * sizeof(double));
  ws->dp = (double *) malloc(n * sizeof(double));
  ws->Q_map = (int *) malloc(n * sizeof(int));
  ws->P_map = (int *) malloc(n * sizeof(int));
  ws->Q_tilde = (int *) malloc(n * sizeof(int));
  ws->c = (int *) malloc(n * sizeof(int));
  ws->log_int = (double *) malloc((n+1) * sizeof(double));
  ws->ratio = (double *) calloc((n+1) * (n+1), sizeof(double));
  ws->plogp = (double *) calloc((n+1) * (n+1), sizeof(double));
  ws->cumhist_data = (int *) malloc(n * n * sizeof(int));
  ws->I_data = (double *) malloc((n+1) * (n+1) * sizeof(double));
  ws->HP2Q_data = (double *) malloc((n+1) * (n+1) * sizeof(double));
  ws->cumhist = (int **) malloc(n * sizeof(int *));
  ws->I = (double **) malloc((n+1) * sizeof(double *));
  ws->HP2Q = (double **) malloc((n+1) * sizeof(double *));

  if (ws->score == NULL || ws->M_temp == NULL || ws->dp == NULL ||
      ws->Q_map == NULL || ws->P_map == NULL || ws->Q_tilde == NULL ||
      ws->c == NULL || ws->log_int == NULL || ws->ratio == NULL ||
      ws->plogp == NULL || ws->cumhist_data == NULL ||
      ws->I_data == NULL || ws->HP2Q_data == NULL ||
      ws->cumhist == NULL || ws->I == NULL || ws->HP2Q == NULL)
    goto error;

  for (i=0; i<n; i++)
    ws->cumhist[i] = &ws->cumhist_data[i*n];
  ws->log_int[0] = 0.0;
  for (i=1; i<=n; i++)
    ws->log_int[i] = log((double) i);
  for (i=1; i<=n; i++)
    {
      int j;
      for (j=1; j<=i; j++)
        {
          size_t index;
          index = (size_t) j*(n+1)+i;
          ws->ratio[index] = (double) j / (double) i;
          ws->plogp[index] = ws->ratio[index] *
            (ws->log_int[j] - ws->log_int[i]);
        }
    }
  for (i=0; i<=n; i++)
    {
      ws->I[i] = &ws->I_data[i*(n+1)];
      ws->HP2Q[i] = &ws->HP2Q_data[i*(n+1)];
    }
  return ws;

 error:
  if (ws->score != NULL)
    mine_free_score(&ws->score);
  free(ws->HP2Q);
  free(ws->I);
  free(ws->cumhist);
  free(ws->HP2Q_data);
  free(ws->I_data);
  free(ws->cumhist_data);
  free(ws->plogp);
  free(ws->ratio);
  free(ws->log_int);
  free(ws->c);
  free(ws->Q_tilde);
  free(ws->P_map);
  free(ws->Q_map);
  free(ws->dp);
  free(ws->M_temp);
  free(ws);
  return NULL;
}


static void free_mic_workspace(mine_mic_workspace *ws)
{
  if (ws == NULL)
    return;
  mine_free_score(&ws->score);
  free(ws->HP2Q);
  free(ws->I);
  free(ws->cumhist);
  free(ws->HP2Q_data);
  free(ws->I_data);
  free(ws->cumhist_data);
  free(ws->plogp);
  free(ws->ratio);
  free(ws->log_int);
  free(ws->c);
  free(ws->Q_tilde);
  free(ws->P_map);
  free(ws->Q_map);
  free(ws->dp);
  free(ws->M_temp);
  free(ws);
}


static int GetSuperclumpsPartition_ws(double *dx, int n, int k_hat,
                                      int *Q_map, int has_ties,
                                      int *P_map, int *p,
                                      mine_mic_workspace *ws)
{
  int i, j, flag, c, s;

  if (has_ties)
    {
      i = 0;
      c = -1;
      memcpy(ws->Q_tilde, Q_map, n*sizeof(int));
      while (i < n)
        {
          s = 1;
          flag = FALSE;
          for (j=i+1; j<n; j++)
            {
              if (dx[i] == dx[j])
                {
                  if (ws->Q_tilde[i] != ws->Q_tilde[j])
                    flag = TRUE;
                  ++s;
                }
              else
                break;
            }
          if ((s > 1) && (flag == TRUE))
            {
              for (j=0; j<s; j++)
                ws->Q_tilde[i+j] = c;
              --c;
            }
          i += s;
        }
    }

  i = 0;
  P_map[0] = 0;
  for (j=1; j<n; j++)
    {
      if ((has_ties ? ws->Q_tilde[j] : Q_map[j]) !=
          (has_ties ? ws->Q_tilde[j-1] : Q_map[j-1]))
        ++i;
      P_map[j] = i;
    }
  *p = i + 1;

  if (*p > k_hat)
    {
      for (i=0; i<n; i++)
        ws->dp[i] = (double) P_map[i];
      EquipartitionYAxis(ws->dp, n, k_hat, P_map, p);
    }
  return 0;
}


static double hq_ws(int **cumhist, int q, int p, int n,
                    mine_mic_workspace *ws)
{
  int i;
  double H = 0.0;

  if (q == 2 || q == 3)
    {
      int count;
      count = cumhist[0][p-1];
      if (count != 0)
        H -= ws->plogp[(size_t) count*(ws->n+1)+n];
      count = cumhist[1][p-1];
      if (count != 0)
        H -= ws->plogp[(size_t) count*(ws->n+1)+n];
      if (q == 3)
        {
          count = cumhist[2][p-1];
          if (count != 0)
            H -= ws->plogp[(size_t) count*(ws->n+1)+n];
        }
      return H;
    }
  for (i=0; i<q; i++)
    {
      int count;
      count = cumhist[i][p-1];
      if (count != 0)
        H -= ws->plogp[(size_t) count*(ws->n+1)+n];
    }
  return H;
}


static double hp3_ws(int *c, int s, int t, mine_mic_workspace *ws)
{
  int sum, total;
  double H = 0.0;

  if (s == t)
    return 0.0;
  total = c[t-1];
  if (c[s-1] != 0)
    H -= ws->plogp[(size_t) c[s-1]*(ws->n+1)+total];
  sum = c[t-1] - c[s-1];
  if (sum != 0)
    H -= ws->plogp[(size_t) sum*(ws->n+1)+total];
  return H;
}


static double hp3q_ws(int **cumhist, int *c, int q, int p, int s, int t,
                      mine_mic_workspace *ws)
{
  int i, sum, total;
  double H = 0.0;

  total = c[t-1];
  if (q == 2 || q == 3)
    {
      int count;
      count = cumhist[0][s-1];
      if (count != 0)
        H -= ws->plogp[(size_t) count*(ws->n+1)+total];
      sum = cumhist[0][t-1] - count;
      if (sum != 0)
        H -= ws->plogp[(size_t) sum*(ws->n+1)+total];
      count = cumhist[1][s-1];
      if (count != 0)
        H -= ws->plogp[(size_t) count*(ws->n+1)+total];
      sum = cumhist[1][t-1] - count;
      if (sum != 0)
        H -= ws->plogp[(size_t) sum*(ws->n+1)+total];
      if (q == 3)
        {
          count = cumhist[2][s-1];
          if (count != 0)
            H -= ws->plogp[(size_t) count*(ws->n+1)+total];
          sum = cumhist[2][t-1] - count;
          if (sum != 0)
            H -= ws->plogp[(size_t) sum*(ws->n+1)+total];
        }
      return H;
    }
  for (i=0; i<q; i++)
    {
      int count;
      count = cumhist[i][s-1];
      if (count != 0)
        H -= ws->plogp[(size_t) count*(ws->n+1)+total];
      sum = cumhist[i][t-1] - cumhist[i][s-1];
      if (sum != 0)
        H -= ws->plogp[(size_t) sum*(ws->n+1)+total];
    }
  return H;
}


static double hp2q_ws(int **cumhist, int *c, int q, int p, int s, int t,
                      mine_mic_workspace *ws)
{
  int i, sum, total_count;
  double H = 0.0;

  if (s == t)
    return 0.0;
  total_count = c[t-1] - c[s-1];
  if (q == 2 || q == 3)
    {
      sum = cumhist[0][t-1] - cumhist[0][s-1];
      if (sum != 0)
        H -= ws->plogp[(size_t) sum*(ws->n+1)+total_count];
      sum = cumhist[1][t-1] - cumhist[1][s-1];
      if (sum != 0)
        H -= ws->plogp[(size_t) sum*(ws->n+1)+total_count];
      if (q == 3)
        {
          sum = cumhist[2][t-1] - cumhist[2][s-1];
          if (sum != 0)
            H -= ws->plogp[(size_t) sum*(ws->n+1)+total_count];
        }
      return H;
    }
  for (i=0; i<q; i++)
    {
      sum = cumhist[i][t-1] - cumhist[i][s-1];
      if (sum != 0)
        H -= ws->plogp[(size_t) sum*(ws->n+1)+total_count];
    }
  return H;
}


static int OptimizeXAxis_ws(int n, int *Q_map, int q, int *P_map, int p,
                            int x, double *score,
                            mine_mic_workspace *ws)
{
  int i, j, s, t, l;
  double F, F_max, HQ;

  if (p == 1)
    {
      for (i=0; i<x-1; i++)
        score[i] = 0.0;
      return 0;
    }

  for (i=0; i<p; i++)
    ws->c[i] = 0;
  for (i=0; i<n; i++)
    ws->c[P_map[i]]++;
  for (i=1; i<p; i++)
    ws->c[i] += ws->c[i-1];

  for (i=0; i<q; i++)
    for (j=0; j<p; j++)
      ws->cumhist[i][j] = 0;
  for (i=0; i<n; i++)
    ws->cumhist[Q_map[i]][P_map[i]]++;
  for (i=0; i<q; i++)
    for (j=1; j<p; j++)
      ws->cumhist[i][j] += ws->cumhist[i][j-1];

  if (x >= 3)
    for (t=3; t<=p; t++)
      for (s=2; s<=t; s++)
        ws->HP2Q[s][t] = hp2q_ws(
          ws->cumhist, ws->c, q, p, s, t, ws
        );

  HQ = hq_ws(ws->cumhist, q, p, n, ws);
  if (x == 2)
    {
      t = p;
      F_max = -DBL_MAX;
      for (s=1; s<=t; s++)
        {
          F = hp3_ws(ws->c, s, t, ws) -
            hp3q_ws(
              ws->cumhist, ws->c, q, p, s, t, ws
            );
          if (F > F_max)
            {
              ws->I[t][2] = HQ + F;
              F_max = F;
            }
        }
      score[0] = ws->I[p][2] /
        MIN(ws->log_int[2], ws->log_int[q]);
      return 0;
    }

  for (t=2; t<=p; t++)
    {
      F_max = -DBL_MAX;
      for (s=1; s<=t; s++)
        {
          F = hp3_ws(ws->c, s, t, ws) -
            hp3q_ws(
              ws->cumhist, ws->c, q, p, s, t, ws
            );
          if (F > F_max)
            {
              ws->I[t][2] = HQ + F;
              F_max = F;
            }
        }
    }

  for (l=3; l<=x; l++)
    {
      for (t=l; t<=p; t++)
        {
          F_max = -DBL_MAX;
          for (s=l-1; s<=t; s++)
            {
              F = (ws->ratio[(size_t) ws->c[s-1]*(ws->n+1)+
                             ws->c[t-1]] *
                   (ws->I[s][l-1]-HQ)) -
                (ws->ratio[(size_t) (ws->c[t-1]-ws->c[s-1])*
                           (ws->n+1)+ws->c[t-1]] *
                 ws->HP2Q[s][t]);
              if (F > F_max)
                {
                  ws->I[t][l] = HQ + F;
                  F_max = F;
                }
            }
        }
    }

  for (i=p+1; i<=x; i++)
    ws->I[p][i] = ws->I[p][p];
  for (i=2; i<=x; i++)
    score[i-2] = ws->I[p][i] /
      MIN(ws->log_int[i], ws->log_int[q]);
  return 0;
}


static double compute_mic_preordered(double *xx, double *yy,
                                     int *ix, int *iy, int n,
                                     int *x_equip_maps, int *y_equip_maps,
                                     int *x_equip_q, int *y_equip_q,
                                     int x_has_ties, int y_has_ties,
                                     mine_parameter *param,
                                     mine_mic_workspace *ws)
{
  int i, j, k, p, q;
  mine_score *score;

  score = ws->score;
  for (i=0; i<score->n; i++)
    {
      k = MAX((int) (param->c * (score->m[i]+1)), 1);
      q = y_equip_q[i];
      for (j=0; j<n; j++)
        ws->Q_map[j] = y_equip_maps[i*n+ix[j]];
      GetSuperclumpsPartition_ws(
        xx, n, k, ws->Q_map, x_has_ties, ws->P_map, &p, ws
      );
      if (param->est == EST_MIC_APPROX)
        OptimizeXAxis_ws(
          n, ws->Q_map, q, ws->P_map, p,
          score->m[i]+1, score->M[i], ws
        );
      else
        OptimizeXAxis_ws(
          n, ws->Q_map, q, ws->P_map, p,
          MIN(i+2, score->m[i]+1), score->M[i], ws
        );
    }

  for (i=0; i<score->n; i++)
    {
      k = MAX((int) (param->c * (score->m[i]+1)), 1);
      q = x_equip_q[i];
      for (j=0; j<n; j++)
        ws->Q_map[j] = x_equip_maps[i*n+iy[j]];
      GetSuperclumpsPartition_ws(
        yy, n, k, ws->Q_map, y_has_ties, ws->P_map, &p, ws
      );
      if (param->est == EST_MIC_APPROX)
        OptimizeXAxis_ws(
          n, ws->Q_map, q, ws->P_map, p,
          score->m[i]+1, ws->M_temp, ws
        );
      else
        OptimizeXAxis_ws(
          n, ws->Q_map, q, ws->P_map, p,
          MIN(i+2, score->m[i]+1), ws->M_temp, ws
        );

      if (param->est == EST_MIC_APPROX)
        for (j=0; j<score->m[i]; j++)
          score->M[j][i] = MAX(ws->M_temp[j], score->M[j][i]);
      else
        for (j=0; j<MIN(i+1, score->m[i]); j++)
          score->M[j][i] = ws->M_temp[j];
    }
  return mine_mic(score);
}


typedef struct mine_mic_precomputed
{
  int score_n;
  double *sorted;
  int *orders;
  int *equip_maps;
  int *equip_q;
  int *has_ties;
} mine_mic_precomputed;


typedef struct mine_filtered_buffer
{
  int *rows;
  int *cols;
  double *mic;
  size_t size;
  size_t capacity;
  int failed;
} mine_filtered_buffer;


static void free_mic_precomputed(mine_mic_precomputed *precomputed)
{
  if (precomputed == NULL)
    return;
  free(precomputed->has_ties);
  free(precomputed->equip_q);
  free(precomputed->equip_maps);
  free(precomputed->sorted);
  free(precomputed->orders);
  free(precomputed);
}


static mine_mic_precomputed *prepare_mic_rows(
  mine_matrix *X, mine_parameter *param, int n_threads
)
{
  int i, failed;
  mine_problem shape_prob;
  mine_score *shape_score;
  mine_mic_precomputed *precomputed;

  precomputed = (mine_mic_precomputed *) calloc(
    1, sizeof(mine_mic_precomputed)
  );
  if (precomputed == NULL)
    return NULL;

  shape_prob.n = X->m;
  shape_score = init_score(&shape_prob, param);
  if (shape_score == NULL)
    {
      free_mic_precomputed(precomputed);
      return NULL;
    }
  precomputed->score_n = shape_score->n;
  mine_free_score(&shape_score);

  precomputed->orders = (int *) malloc(
    (size_t) X->n * X->m * sizeof(int)
  );
  precomputed->sorted = (double *) malloc(
    (size_t) X->n * X->m * sizeof(double)
  );
  precomputed->equip_maps = (int *) malloc(
    (size_t) X->n * precomputed->score_n * X->m * sizeof(int)
  );
  precomputed->equip_q = (int *) malloc(
    (size_t) X->n * precomputed->score_n * sizeof(int)
  );
  precomputed->has_ties = (int *) malloc((size_t) X->n * sizeof(int));
  if (precomputed->orders == NULL || precomputed->sorted == NULL ||
      precomputed->equip_maps == NULL || precomputed->equip_q == NULL ||
      precomputed->has_ties == NULL)
    {
      free_mic_precomputed(precomputed);
      return NULL;
    }

  if (n_threads <= 0)
    n_threads = 1;

  failed = FALSE;
  #pragma omp parallel for schedule(static) num_threads(n_threads)
  for (i=0; i<X->n; i++)
    {
      int j;
      int *idx;
      int *equip_sorted;
      idx = argsort(&X->data[i*X->m], X->m);
      equip_sorted = (int *) malloc((size_t) X->m * sizeof(int));
      if (idx == NULL || equip_sorted == NULL)
        {
          #pragma omp atomic write
          failed = TRUE;
          free(idx);
          free(equip_sorted);
        }
      else
        {
          precomputed->has_ties[i] = FALSE;
          for (j=0; j<X->m; j++)
            {
              precomputed->orders[i*X->m+j] = idx[j];
              precomputed->sorted[i*X->m+j] =
                X->data[i*X->m+idx[j]];
              if (j > 0 &&
                  precomputed->sorted[i*X->m+j] ==
                  precomputed->sorted[i*X->m+j-1])
                precomputed->has_ties[i] = TRUE;
            }
          for (j=0; j<precomputed->score_n; j++)
            {
              int k;
              EquipartitionYAxis(
                &precomputed->sorted[i*X->m], X->m, j+2,
                equip_sorted,
                &precomputed->equip_q[i*precomputed->score_n+j]
              );
              for (k=0; k<X->m; k++)
                precomputed->equip_maps[
                  (i*precomputed->score_n+j)*X->m+idx[k]
                ] = equip_sorted[k];
            }
          free(equip_sorted);
          free(idx);
        }
    }
  if (failed)
    {
      free_mic_precomputed(precomputed);
      return NULL;
    }
  return precomputed;
}


static double compute_prepared_pair(
  mine_matrix *X, mine_parameter *param,
  mine_mic_precomputed *precomputed, mine_mic_workspace *ws,
  int i, int j
)
{
  return compute_mic_preordered(
    &precomputed->sorted[i*X->m], &precomputed->sorted[j*X->m],
    &precomputed->orders[i*X->m], &precomputed->orders[j*X->m],
    X->m,
    &precomputed->equip_maps[i*precomputed->score_n*X->m],
    &precomputed->equip_maps[j*precomputed->score_n*X->m],
    &precomputed->equip_q[i*precomputed->score_n],
    &precomputed->equip_q[j*precomputed->score_n],
    precomputed->has_ties[i], precomputed->has_ties[j],
    param, ws
  );
}


static int append_filtered_value(
  mine_filtered_buffer *buffer, int row, int col, double mic
)
{
  if (buffer->size == buffer->capacity)
    {
      size_t new_capacity;
      int *new_rows;
      int *new_cols;
      double *new_mic;

      new_capacity = buffer->capacity == 0 ? 1024 : buffer->capacity * 2;
      new_rows = (int *) realloc(
        buffer->rows, new_capacity * sizeof(int)
      );
      if (new_rows == NULL)
        goto error;
      buffer->rows = new_rows;

      new_cols = (int *) realloc(
        buffer->cols, new_capacity * sizeof(int)
      );
      if (new_cols == NULL)
        goto error;
      buffer->cols = new_cols;

      new_mic = (double *) realloc(
        buffer->mic, new_capacity * sizeof(double)
      );
      if (new_mic == NULL)
        goto error;
      buffer->mic = new_mic;
      buffer->capacity = new_capacity;
    }

  buffer->rows[buffer->size] = row;
  buffer->cols[buffer->size] = col;
  buffer->mic[buffer->size] = mic;
  buffer->size++;
  return 0;

 error:
  buffer->failed = TRUE;
  return -1;
}


static void free_filtered_buffers(
  mine_filtered_buffer *buffers, int n_buffers
)
{
  int i;
  if (buffers == NULL)
    return;
  for (i=0; i<n_buffers; i++)
    {
      free(buffers[i].rows);
      free(buffers[i].cols);
      free(buffers[i].mic);
    }
  free(buffers);
}


double *mine_compute_mic_pstats(mine_matrix *X, mine_parameter *param,
                                int n_threads)
{
  int i, failed;
  long long n_pairs;
  double *mic;
  mine_mic_precomputed *precomputed;

  n_pairs = ((long long) X->n * (X->n - 1)) / 2;
  mic = (double *) malloc((size_t) n_pairs * sizeof(double));
  if (mic == NULL)
    return NULL;
  if (n_threads <= 0)
    n_threads = 1;

  precomputed = prepare_mic_rows(X, param, n_threads);
  if (precomputed == NULL)
    {
      free(mic);
      return NULL;
    }

  failed = FALSE;
  #pragma omp parallel num_threads(n_threads)
  {
    mine_mic_workspace *ws;
    ws = init_mic_workspace(X->m, param);
    if (ws == NULL)
      {
        #pragma omp atomic write
        failed = TRUE;
      }

    #pragma omp for schedule(dynamic, 1)
    for (i=0; i<X->n-1; i++)
      {
        int j;
        if (ws == NULL)
          continue;
        for (j=i+1; j<X->n; j++)
          {
            long long k;
            k = (long long) X->n*i -
              ((long long) i*(i+1))/2 - i - 1 + j;
            mic[k] = compute_prepared_pair(
              X, param, precomputed, ws, i, j
            );
          }
      }
    free_mic_workspace(ws);
  }

  free_mic_precomputed(precomputed);
  if (failed)
    {
      free(mic);
      return NULL;
    }
  return mic;
}


double *mine_compute_mic_pairs(mine_matrix *X, int *rows, int *cols,
                               long long n_pairs, mine_parameter *param,
                               int n_threads)
{
  int failed;
  long long k;
  double *mic;
  mine_mic_precomputed *precomputed;

  mic = (double *) malloc(
    (size_t) (n_pairs > 0 ? n_pairs : 1) * sizeof(double)
  );
  if (mic == NULL)
    return NULL;
  if (n_threads <= 0)
    n_threads = 1;

  precomputed = prepare_mic_rows(X, param, n_threads);
  if (precomputed == NULL)
    {
      free(mic);
      return NULL;
    }

  failed = FALSE;
  #pragma omp parallel num_threads(n_threads)
  {
    mine_mic_workspace *ws;
    ws = init_mic_workspace(X->m, param);
    if (ws == NULL)
      {
        #pragma omp atomic write
        failed = TRUE;
      }

    #pragma omp for schedule(dynamic, 1)
    for (k=0; k<n_pairs; k++)
      {
        int i, j;
        if (ws == NULL)
          continue;
        i = rows[k];
        j = cols[k];
        if (i < 0 || j < 0 || i >= X->n || j >= X->n || i == j)
          {
            #pragma omp atomic write
            failed = TRUE;
            continue;
          }
        mic[k] = compute_prepared_pair(
          X, param, precomputed, ws, i, j
        );
      }
    free_mic_workspace(ws);
  }

  free_mic_precomputed(precomputed);
  if (failed)
    {
      free(mic);
      return NULL;
    }
  return mic;
}


mine_filtered_pstats *mine_compute_mic_pstats_filtered(
  mine_matrix *X, mine_parameter *param, int n_threads, double min_mic,
  int row_start, int row_end
)
{
  int i, failed, buffer_failed;
  int n_buffers;
  long long total, offset;
  mine_filtered_buffer *buffers;
  mine_filtered_pstats *result;
  mine_mic_precomputed *precomputed;

  if (row_start < 0 || row_end < row_start || row_end > X->n-1)
    return NULL;
  if (n_threads <= 0)
    n_threads = 1;
  n_buffers = n_threads;
  buffers = (mine_filtered_buffer *) calloc(
    (size_t) n_buffers, sizeof(mine_filtered_buffer)
  );
  result = (mine_filtered_pstats *) calloc(
    1, sizeof(mine_filtered_pstats)
  );
  if (buffers == NULL || result == NULL)
    goto error_without_precomputed;

  precomputed = prepare_mic_rows(X, param, n_threads);
  if (precomputed == NULL)
    goto error_without_precomputed;

  failed = FALSE;
  #pragma omp parallel num_threads(n_threads)
  {
    int thread_id;
    mine_mic_workspace *ws;
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#else
    thread_id = 0;
#endif
    ws = init_mic_workspace(X->m, param);
    if (ws == NULL)
      {
        #pragma omp atomic write
        failed = TRUE;
      }

    #pragma omp for schedule(dynamic, 1)
    for (i=row_start; i<row_end; i++)
      {
        int j;
        if (ws == NULL || buffers[thread_id].failed)
          continue;
        for (j=i+1; j<X->n; j++)
          {
            double value;
            value = compute_prepared_pair(
              X, param, precomputed, ws, i, j
            );
            if (value >= min_mic &&
                append_filtered_value(
                  &buffers[thread_id], i, j, value
                ) != 0)
              {
                #pragma omp atomic write
                failed = TRUE;
                break;
              }
          }
      }
    free_mic_workspace(ws);
  }

  buffer_failed = FALSE;
  total = 0;
  for (i=0; i<n_buffers; i++)
    {
      if (buffers[i].failed)
        buffer_failed = TRUE;
      total += (long long) buffers[i].size;
    }
  if (failed || buffer_failed)
    goto error;

  result->rows = (int *) malloc(
    (size_t) (total > 0 ? total : 1) * sizeof(int)
  );
  result->cols = (int *) malloc(
    (size_t) (total > 0 ? total : 1) * sizeof(int)
  );
  result->mic = (double *) malloc(
    (size_t) (total > 0 ? total : 1) * sizeof(double)
  );
  if (result->rows == NULL || result->cols == NULL ||
      result->mic == NULL)
    goto error;

  offset = 0;
  for (i=0; i<n_buffers; i++)
    {
      size_t count;
      count = buffers[i].size;
      memcpy(&result->rows[offset], buffers[i].rows, count * sizeof(int));
      memcpy(&result->cols[offset], buffers[i].cols, count * sizeof(int));
      memcpy(&result->mic[offset], buffers[i].mic, count * sizeof(double));
      offset += (long long) count;
    }
  result->n = total;

  free_mic_precomputed(precomputed);
  free_filtered_buffers(buffers, n_buffers);
  return result;

 error:
  free_mic_precomputed(precomputed);
 error_without_precomputed:
  free_filtered_buffers(buffers, n_buffers);
  if (result != NULL)
    {
      free(result->rows);
      free(result->cols);
      free(result->mic);
      free(result);
    }
  return NULL;
}


mine_cstats *mine_compute_cstats(mine_matrix *X, mine_matrix *Y,
                                 mine_parameter *param)
{
  int i, j, k;
  mine_cstats *stats;
  mine_problem prob;
  mine_score *score;


  if (X->m != Y->m)
    return NULL;

  /* Allocate memory for stats */
  stats = (mine_cstats *) malloc(sizeof(mine_cstats));
  stats->n = X->n;
  stats->m = Y->n;
  stats->mic = (double *) malloc((stats->n * stats->m) * sizeof(double));
  stats->tic = (double *) malloc((stats->n * stats->m) * sizeof(double));

  k = 0;
  prob.n = X->m;
  for (i=0; i<X->n; i++)
    {
      prob.x = &X->data[i*X->m];

      for (j=0; j<Y->n; j++)
        {
          prob.y = &Y->data[j*Y->m];
          score = mine_compute_score(&prob, param);
          stats->mic[k] = mine_mic(score);
          stats->tic[k] = mine_tic(score, TRUE);
          mine_free_score(&score);
          k++;
        }
    }

  return stats;
}
