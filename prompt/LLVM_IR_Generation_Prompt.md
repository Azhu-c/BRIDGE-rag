### Lift the source assembly code into LLVM IR with no optimization(O0), using the provided asm-IR pair example as a functional reference.

### Example 1:
Assembly:'''asm {asm}'''
LLVM IR:'''llvm {llvm ir}'''

### Example 2:
Assembly:'''asm {asm}'''
LLVM IR:'''llvm {llvm ir}'''

### Example 3:
Assembly:'''asm {asm}'''
LLVM IR:'''llvm {llvm ir}'''

### Lift the source assembly code into LLVM IR func using the example,Output the corrected LLVM IR code with the necessary changes, without any commentary. 

 - Use proper LLVM IR syntax and SSA form.Variable names should avoid
   using numbers when possible.
 - Preserve explicit control flow and variable types.
 - Do not apply compiler-level optimizations (e.g., loop unrolling, instruction combining).

### Soure Assembly：
 '''asm
{asm_input}
'''




