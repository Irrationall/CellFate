#include <stdint.h>
#include <stddef.h>

/* Independent column accumulators; CSR entries retain their original order. */
void csr_product16(int64_t n, const double *restrict data,
                   const int32_t *restrict indices, const int32_t *restrict indptr,
                   const double *restrict current, double *restrict target)
{
    for (int64_t i = 0; i < n; ++i) {
        double a0=0., a1=0., a2=0., a3=0., a4=0., a5=0., a6=0., a7=0.;
        double a8=0., a9=0., a10=0., a11=0., a12=0., a13=0., a14=0., a15=0.;
        for (int32_t p = indptr[i]; p < indptr[i+1]; ++p) {
            const double v = data[p];
            const double *row = current + (size_t)indices[p] * 16;
            a0 += v * row[0];
            a1 += v * row[1];
            a2 += v * row[2];
            a3 += v * row[3];
            a4 += v * row[4];
            a5 += v * row[5];
            a6 += v * row[6];
            a7 += v * row[7];
            a8 += v * row[8];
            a9 += v * row[9];
            a10 += v * row[10];
            a11 += v * row[11];
            a12 += v * row[12];
            a13 += v * row[13];
            a14 += v * row[14];
            a15 += v * row[15];
        }
        double *out = target + (size_t)i * 16;
        out[0]=a0; out[1]=a1; out[2]=a2; out[3]=a3;
        out[4]=a4; out[5]=a5; out[6]=a6; out[7]=a7;
        out[8]=a8; out[9]=a9; out[10]=a10; out[11]=a11;
        out[12]=a12; out[13]=a13; out[14]=a14; out[15]=a15;
    }
}
