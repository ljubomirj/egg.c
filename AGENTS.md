# AGENTS

## Repository Guidelines

This repository implements egg - the ML training algorithm as descibed in

Evolution Strategies at the Hyperscale
https://eshyperscale.github.io/
https://www.alphaxiv.org/abs/2511.16652
https://github.com/ESHyperscale/HyperscaleES
https://github.com/ESHyperscale/nano-egg

## GPT-5.1 Codex (this workspace)
- **Role:** Pair-programming autopilot focused on C/C++/Metal reinforcement-learning code in `egg.c`.
- **Strengths:** 
  - Understands and modifies large single-file C projects.
  - Comfortable wiring GPU backends (Metal now, ROCm/CUDA soon) alongside existing CPU flows.
  - Keeps Makefile/README/docs in sync with code changes.
- **Workflow Tips:**
  1. Pin down the target build (`egg`, `egg-cpumulti`, `egg-gpumetal`, etc.) before asking for changes.
  2. Mention whether we’re in *ask* (read-only) or *agent* mode so I know if execution is allowed.
  3. For long-running training runs, specify whether to let them finish or stop early once startup is verified.
  4. If you need reproducible behavior, give me the dataset slice, seed, and build flags so GPU/CPU paths can be compared.
- **Handoff:** When you plan to continue edits manually, ask for a quick summary or TODO list; I’ll outline remaining steps and any caveats (e.g., partially migrated kernels).

## Coding agents cooperation

Write into README.md or task.md or project.md as you deem fit.

I myself Ljubomir - LJ, I'm writing into README.LJ. Do not write into README.LJ yourself. Never write into README.LJ. But feel free to read it as will. But leave the writing into README.LJ to LJ.

## LJ C/C++ style guide

### Conventions, rules of thumb.

The inspirational book for this was "Writing Solid Code" by Steve Maguire.

Break them for a reason & within reason, otherwise stick to them. When weighting alternatives consider code reading the highest priority. 
Make it easy to reason at the point of use, save the reader the effort of jumping to the definition.
(comptime = compile time, compile-time; runtime = run time, run-time)

1. If code needs Stackoverflow search and language layering to be understood, then it is no good: change it until it is blindingly obvious what is supposed to happen.
  Boring, bog standard - good. Dynamic and exciting - bad.

2. READABILITY IS THE STEPPING STONE TO CORRECTNESS: 1st understand what needs to happen, 2nd then assert, check etc that it does indeed happen.

3. Use const & for costly-to-copy types input arguments. Use pointers when arguments are changed (*not* non-const references) for output and input/output arguments.
  Readability 1st - have the reader of the calling code know the variable changes in the function.

4. Use explicit in constructors with non-trivial arguments.

5. C++ header fiels are .hpp, while C header files are .h. Do NOT put C++ code in .h files - stick to .h/.hpp for C/C++ respectivelly.
  Support common headers prefix file Prefix.h or target specific one, in the project headers and the Makefile.

6. Public functions and classes upper cammel case, private lower cammel or better classic C snake case - easier to distinguish for a reader. 
  Member variables prefixed with "m", input args with "i", output arg with "o", input-output args "io".
  Break the rule for 1st argument of function that actually operates on an "object" analogous to "this" in C++, e.g it is ok to have:
  XmapXyz(Xmap *xm), XfileXyz(Xfile *xf), XyzPrint(FILE *f).

7. Single line comments can end without fullstop, multilines punctuated.

8. Preproc use #if, not #ifdef if possible, makes explicit the case of #define BLAH 0. But use #ifdef MSC_VER.

9. Pointer declaration is "T *p" not "T* p" nor "T * p". The * binds with p stronger than with T. 
   However for references use "T & v" instead of "T &v" or "T& v" so not to miss the reference and aid readability.

10. Keep the "business code" in cammel case as per recommendations. The non-"business code", the scaffolding orthogonal to it, e.g
  verbose logging, assertions, checking of runtime errors, have it in upper case e.g CHECK XASSERT V::P("logging msg"), so
  to be able to mentally ignore tune out when reading the "business logic". READABILITY ENABLES CORRECTNESS.

11. Variable names in cammel case, with type info not part of the name. Make exception for complicated types like specific matrix
  sizes, e.g Matrix2d data2_R2_XY(numRows2, sizeX + sizeY); or index into some array of elements, e.g long sec_index = oSyms->GetIndex(p);
  or int fitDate_index = SnapLeft2<long>(iFitDate, iUni.mNumDayCal, iUni.mDayCal).

12. A vector operation done through array iteration for(i=0...){sth[i];} is ok in 1 line - just enclose the body in {}, even if a single line.

13. Prefer prefix operators over postfix (i.e ++i over i++) for weak reasons: standardization, easier reading ("increment i" over "i incremented").

For a "C" class Abc:

14. AbcNew does alloc + init.

15. AbcInit does init only.

"C" class Def extending Abc can:

16. in DefNew do AbcNew (which does alloc and init of Abc) + own alloc + own init via DefInit.

17. in DefInit do init only, not touching Abc.

18. Simlarly, AbcDelete does shutdown + free, while AbcShutdown does shutdown only.
  Then DefDelete can do AbcDelete + DefShutdown + free, while DefShutdown does Def shutdown only.

For Vector-s and Matrix-es naming conventions:

19. Names are lower letters like all other variables. Admitedly it is tempting to capitalize, but don't.

20. Do have the dimensions in short hand single letter capital affixed to the name, in a way to give the reader some idea (even if incomplete and not 100% clear) about the size of the matrix - e.g:
   Matrix data_R_XY(numRows, sizeX + sizeY);

21. Prefer "using" to "typedef" for the classes of matrix templates, e.g:
   template class Matrix<double>;      using Matrix2d = Matrix<double>;      using AutoMatrix2d = AutoPtr<Matrix2d>;

22. In the specific classes to templates affix the dimensionality and the type of the element at end of name, e.g:
   template class Vector<long>;        using Vector1l = Vector<long>;        using AutoVector1l = AutoPtr<Vector1l>;
   template class Vector<char *>;      using Vector1pc = Vector<char *>;     using AutoVector1pc = AutoPtr<Vector1pc>;

For Matrix classes that have separate slicing objects like MatrixRef, esp the Pmap ones:

23. Affix _ref to the slicing object, like this PmapMatrixRef<T> myMx_ref = myMx.GetRef() don't mixup with the name in cammel case.

24. Result of slicing operation also affixed same way PmapVectorRef<T> myMx_ref_i = myMx_ref[i] so clear where it came from and
   separate from the name. All further derivatives have _deriv affixed to the name, in order of derivation. So keeps name separate,
   and readible where it came from.

For mmap-ed classes:

25. Keep the (small) metadata separate from the (big) mmaped data. Data in mmap can be easily accessed from matlab, numpy, R.
   Having meta data mixed in with the data would make that difficult. Metadata can be hard coded by the user, or in a separate file.
   It is up to the user how metadata should be persisted. In runtime metadata can be assumed to be in memory.

26. Have metadata members declared 1st at the top of the class, so to be able to quickly see what needs to be de/serialized.

For old style C arrays use notation:

27. Allocated space is xyzSize const mNameSize=NUM_ELEMENTS; define mName[mNameSize]. If static array known at comptime then ok to leave out 
   xyzSize use XNUMEL(mName) instead of mNameSize. XNUMEL is pointer-safe passing pointer instead of array results in comptime error.

28. Number of elements present is xyzCount e.g define mNameCount=0, assign add element mName[mNameCount++]=.

29. Do not use "number of" Num notation as mNumName or mNameNum as it is ambiguous (does "number" refer to the "size" or the "count"). 
  Use Size and Count as suffix to make it explicit which one it is. Keep Num for genuine counters that are not refering to allocated size.

30. Encode matrix dimensions as zzz_AxB e.g. mImbalance_NxBxS, mPosition_TxS, mClosePrc_NxS. Keep even the 1s in vectors, e.g. mWei_1xS, and esp if 
  it comes via slicing mWeight_NxS, e.g. mWei_1xS = mWeight_NxS[i]. With runtime defined matrix size, use U for the unknown runtime dim e.g. for 
  comptime number of predictors P have float Xfit_UxP for matrix [U x P] with U known at runtime, P known at comptime.

31. If vectors are _1xN or _Nx1, use suffix _m or _n to denote m-th or n-th element of something, e.g. scalar xi = X_1xN[i].

32. For dynamic arrays in function arguments, use function(size_t N, T arr[N]) to pass the length of the array at runtime. NB it was the case
  for runtime that was not possible, could only document as T arr[/*N*/]) for the human reader. C matrix via Variably-Modified Types (VMT)

  https://gustedt.wordpress.com/2011/01/09/dont-be-afraid-of-variably-modified-types/, https://blog.joren.ga/vla-usecases:

  #include <stdio.h>
  #include <malloc.h>
  // NB 'n', 'm' and 'b' are runtime variables holding the dimensions, not comptime constants.
  // However, passing void funarr(int m,int n,int b,int(*arr_MxNxB)[m][n][b]){ works in C but not C++ - doh!
  void funarr(int m, int n, int b, int (*arr_MxNxB)/*[m][n][b]*/) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int k = 0; k < b; ++k) {	(*arr_MxNxB)[i][j][k] = i + j + k; }
      }
    }
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int k = 0; k < b; ++k) { printf("arr_MxNxK[%d][%d][%d]=%d expect %d\n", i, j, k, (*arr_MxNxB)[i][j][k], i + j + k); }
      }
    }
  }
  void main() {
    int m = 3, n = 4, b = 5;
    //int (*arr_MxNxB)[m][n][b] = (decltype(arr_MxNxB))malloc(sizeof(*arr_MxNxB)); // decltype requires C++
    int (*arr_MxNxB)[m][n][b] = malloc(sizeof(*arr_MxNxB)); // in C we don't need to cast
    if (arr_MxNxB) {
      funarr(m, n, b, arr_MxNxB);
      free(arr_MxNxB);
    }
    printf("arr_MxNxB m=%d n=%d b=%d sizeof(arr_MxNxB)=%zu sizeof(arr_MxNxB[0][0][0])=%zu\n", m, n, b, sizeof(*arr_MxNxB), sizeof((*arr_MxNxB)[0][0][0]));
  }

33. Virtual functions in SubClass class must have override (for the virtual in SuperClass), but must not have virtual themselves.

Space in templates arguments:

34. In definition add space as usual: template <class T, int N> class Marray : public MarrayBase {

35. In use of templated type assume everything is part of the tupe so do not add space: size_t len1 = Marray<T,N-1>::Sizeof(ap);

Format specs for library types:

36. For size_t is %zu, for ssize_t is %zd

37. For off_t, off64_t and other signed types may not exist, workaround (intmax_t) casts to widest int then use %jd. Alternative is PRI  (and SCA, for scanf) 
   macros in <cinttypes> or <inttypes.h>.

Fields of data are fieldset, often have common datetime line:

36. Keep the concrete field type known at comptime. If not possible, then provide enum/int-to-type comptime map to cast-down any super-class 
   (that abstracts the concrete type).

37. When operating on a fieldset, use local functions + some repetition in preference to 1) macros 2) boilerplate copy & paste & modify.
   Readability is concern #1-2-3, as is a stepping stone to correctness.

Conditional compilation formatting:

38. When conditional compilation with continuation in a new line, one tab space siffices for all continuations of the same line. No need for every 
   conditional to add its own space. E.g. -
   YES -->   some function(a1 arg1,                    NO -->    some function(a1 arg1,
             #if COND1                                           #if COND1
               a2 arg2, a3 arg3,                                   a2 arg2, a3 arg3,
             #endif                                              #endif
               a4 arg4,                                              a4 arg4,
             #if COND2                                           #if COND2
               a5 arg5,                                                a5 arg5,
             #endif                                              #endif
               a6 arg6, a7 arg7) {                                       a6 arg6, a7 arg7) {
             }                                                   }

39. Comptime parametrisations versus runtime, comptime implemented via class templates or preprocessor, class templates and classes.
  Comptime implemented via templated classess results in templates produced classes that to the C++ class system look completely 
  unrelated. This then makes it hard to process all classes produced by a template in an uniform way within a single code block.
  So if there is only ever going to be one realisation of a comptime parameter in one program, stick to the preprocessor. 
  Example of this is NUM_RESP and NUM_HOR - it's unlikely that we will have different values for them in the same binary.
  On the other end - think whether separate value of the parameters will make a new class. Is Alpha with different number of 
  predictors with NUM_PRED in a template completely different class, unrelated in any way to all other template realisations,
  or not? If the answer is "no", then keep the parameter as runtime, not comptime - even when the value is known at comptime.
  Where comptime->runtime replacement replaces T arr[SIZE] = { NAN } with std::vector<T> vec, use vec.resize(size, NAN) to
  a) initialize the vec-tor easily b) dimension vec such that subsequent code vec[i] does not need changing, as vec is 
  already pre-allocated and no need for push_back() or similar to assign values to the vec-tor that replaced the arr-ay.

Feel free to extend this document with additional agents or instructions as the project grows.***

