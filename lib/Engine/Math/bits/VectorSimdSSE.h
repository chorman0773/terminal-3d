
#include <immintrin.h>
#include <smmintrin.h>

typedef struct Vector4{
    __m128 _m_inner;
} Vector4;

typedef struct Vector3{
    __m128 _m_inner;
} Vector3;

inline static float _float_from_int_bits(int x){
    union{
        int x;
        float f;
    } val = {.x = x};

    return val.f;
}

#define VECCOMP(x,n) (_float_from_int_bits(_mm_extract_ps((x)._m_inner,(n))))

#define X 0
#define Y 1
#define Z 2
#define W 3

#define DEF_VECTOR_ARITH_MACRO(ty, op, mm_op)\
    inline static ty op##ty(ty _v0, ty _v1){\
        return (ty){._m_inner = _mm_##mm_op##_ps(_v0._m_inner, _v1._m_inner)};\
    }

DEF_VECTOR_ARITH_MACRO(Vector3,Add, add)
DEF_VECTOR_ARITH_MACRO(Vector4,Add, add)
DEF_VECTOR_ARITH_MACRO(Vector3, Sub, sub)
DEF_VECTOR_ARITH_MACRO(Vector4, Sub, sub)

#define DEF_VECTOR_ARITH_SCALAR_MACRO(ty, op, mm_op)\
    inline static ty op##ty(ty _v0, float _s1){\
        return (ty){._m_inner = _mm_##mm_op##_ps(_v0._m_inner,_mm_set1_ps(_s1))};\
    }

DEF_VECTOR_ARITH_SCALAR_MACRO(Vector3, Mul, mul)
DEF_VECTOR_ARITH_SCALAR_MACRO(Vector4, Mul, mul)

#define DEF_VECTOR_DOTPRODUCT_NORMALIZE(ty, top_mask)\
    inline static float Dot##ty(ty _v0, ty _v1){\
        return _float_from_int_bits(_mm_extract_ps(_mm_dp_ps(_v0,_v1,(top_mask) | 0x01),0));\
    }\
    inline static ty Normalize##ty(ty _v0){\
        float _dp = Dot##ty(_v0,_v0);\
        __m128 _dp_xmm = _mm_set1_ps(_dp);\
        return (ty){._m_inner = _mm_mul_ps(_v0,_mm_rsqrt_ps(_dp_xmm))};\
    }

DEF_VECTOR_DOTPRODUCT_NORMALIZE(Vector3, 0x70)
DEF_VECTOR_DOTPRODUCT_NORMALIZE(Vector4, 0xF0)

#undef DEF_VECTOR_ARITH_MACRO
#undef DEF_VECTOR_ARITH_SCALAR_MACRO
#undef DEF_VECTOR_DOTPRODUCT_NORMALIZE

#define ZerosVector3 ((Vector3){._m_inner = _mm_setzero_ps()})
#define OnesVector3 ((Vector3){._m_inner = _mm_set1_ps(1)})
#define ZerosVector4 ((Vector4){._m_inner = _mm_setzero_ps()})
#define OnesVector4 ((Vector4){._m_inner = _mm_set1_ps(1)})

inline static Vector3 CrossVector3(Vector3 _v0, Vector3 _v1){
    __m128 _v0intr1 = _mm_shuffle_ps(_v0._m_inner,0xC9);
    __m128 _v0intr2 = _mm_shuffle_ps(_v0._m_inner,0xD2);
    __m128 _v1intr1 = _mm_shuffle_ps(_v1._m_inner,0xC9);
    __m128 _v1intr2 = _mm_shuffle_ps(_v1._m_inner,0xD2);

    __m128 _vres1 = _mm_mul_ps(_v0intr1,_v1intr2);
    __m128 _vres2 = _mm_mul_ps(_v1intr1,_v0intr2);

    return (Vector3){._m_inner = _mm_sub_ps(_vres1,_vres2)};
}