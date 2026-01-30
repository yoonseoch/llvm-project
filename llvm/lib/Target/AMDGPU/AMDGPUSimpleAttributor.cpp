//===- AMDGPUSimpleAttributor.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file A minimal CGSCC pass for testing Attributor framework with AMDGPU.
/// This pass only handles AAUniformWorkGroupSize to demonstrate top-down
/// propagation behavior in a CGSCC context.
///
/// Key difference from standard CGSCC passes: we add ALL functions from the
/// module to the Attributor's Functions set, not just the SCC functions.
/// This tests whether giving Attributor full module visibility while running
/// in CGSCC context enables proper top-down propagation.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/Attributor.h"

#define DEBUG_TYPE "amdgpu-simple-attributor"

using namespace llvm;

namespace {

/// AAUniformWorkGroupSize - Propagate uniform-work-group-size attribute from
/// kernel entry functions to device functions they call.
struct AAUniformWorkGroupSize
    : public StateWrapper<BooleanState, AbstractAttribute> {
  using Base = StateWrapper<BooleanState, AbstractAttribute>;
  AAUniformWorkGroupSize(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  /// Create an abstract attribute view for the position \p IRP.
  static AAUniformWorkGroupSize &createForPosition(const IRPosition &IRP,
                                                   Attributor &A);

  /// See AbstractAttribute::getName().
  StringRef getName() const override { return "AAUniformWorkGroupSize"; }

  /// See AbstractAttribute::getIdAddr().
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is
  /// AAUniformWorkGroupSize.
  static bool classof(const AbstractAttribute *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;
};
const char AAUniformWorkGroupSize::ID = 0;

struct AAUniformWorkGroupSizeFunction : public AAUniformWorkGroupSize {
  AAUniformWorkGroupSizeFunction(const IRPosition &IRP, Attributor &A)
      : AAUniformWorkGroupSize(IRP, A) {}

  void initialize(Attributor &A) override {
    Function *F = getAssociatedFunction();
    CallingConv::ID CC = F->getCallingConv();

    if (CC != CallingConv::AMDGPU_KERNEL)
      return;

    bool InitialValue = false;
    if (F->hasFnAttribute("uniform-work-group-size"))
      InitialValue =
          F->getFnAttribute("uniform-work-group-size").getValueAsString() ==
          "true";

    if (InitialValue)
      indicateOptimisticFixpoint();
    else
      indicatePessimisticFixpoint();
  }

  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Change = ChangeStatus::UNCHANGED;

    auto CheckCallSite = [&](AbstractCallSite CS) {
      Function *Caller = CS.getInstruction()->getFunction();
      LLVM_DEBUG(dbgs() << "[AAUniformWorkGroupSize] Call " << Caller->getName()
                        << "->" << getAssociatedFunction()->getName() << "\n");

      const auto *CallerInfo = A.getAAFor<AAUniformWorkGroupSize>(
          *this, IRPosition::function(*Caller), DepClassTy::REQUIRED);
      if (!CallerInfo || !CallerInfo->isValidState())
        return false;

      Change = Change | clampStateAndIndicateChange(this->getState(),
                                                    CallerInfo->getState());

      return true;
    };

    bool AllCallSitesKnown = true;
    if (!A.checkForAllCallSites(CheckCallSite, *this, true, AllCallSitesKnown))
      return indicatePessimisticFixpoint();

    return Change;
  }

  ChangeStatus manifest(Attributor &A) override {
    SmallVector<Attribute, 8> AttrList;
    LLVMContext &Ctx = getAssociatedFunction()->getContext();

    AttrList.push_back(Attribute::get(Ctx, "uniform-work-group-size",
                                      getAssumed() ? "true" : "false"));
    return A.manifestAttrs(getIRPosition(), AttrList,
                           /* ForceReplace */ true);
  }

  bool isValidState() const override {
    // This state is always valid, even when the state is false.
    return true;
  }

  const std::string getAsStr(Attributor *) const override {
    return "AMDWorkGroupSize[" + std::to_string(getAssumed()) + "]";
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}
};

AAUniformWorkGroupSize &
AAUniformWorkGroupSize::createForPosition(const IRPosition &IRP,
                                          Attributor &A) {
  if (IRP.getPositionKind() == IRPosition::IRP_FUNCTION)
    return *new (A.Allocator) AAUniformWorkGroupSizeFunction(IRP, A);
  llvm_unreachable(
      "AAUniformWorkGroupSize is only valid for function position");
}

} // namespace

PreservedAnalyses AMDGPUSimpleAttributorPass::run(LazyCallGraph::SCC &C,
                                                  CGSCCAnalysisManager &AM,
                                                  LazyCallGraph &CG,
                                                  CGSCCUpdateResult &UR) {
  if (C.size() == 0)
    return PreservedAnalyses::all();
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();
  AnalysisGetter AG(FAM);

  // Get module from any function in the SCC
  Module &M = *C.begin()->getFunction().getParent();

  LLVM_DEBUG(dbgs() << "[AMDGPUSimpleAttributor] Processing SCC with "
                    << C.size() << " function(s) in module " << M.getName()
                    << "\n");
  LLVM_DEBUG(for (LazyCallGraph::Node &N : C) {
    dbgs() << "  SCC function: " << N.getFunction().getName() << "\n";
  });

  // KEY: Add ALL functions from module, not just SCC functions.
  // This gives Attributor visibility into all functions including kernels,
  // enabling top-down propagation to work.
  SetVector<Function *> Functions;
#if 1
  for (Function &F : M) {
    if (!F.isIntrinsic())
      Functions.insert(&F);
  }
#else
  // Doesn't work as it needs to update functions outside of the SCC.
  for (LazyCallGraph::Node &N : C) {
    Function *Fn = &N.getFunction();
    if (!Fn->isIntrinsic())
      Functions.insert(Fn);
  }
#endif

  LLVM_DEBUG(dbgs() << "[AMDGPUSimpleAttributor] Added " << Functions.size()
                    << " functions from module to Attributor\n");

  if (Functions.empty())
    return PreservedAnalyses::all();

  CallGraphUpdater CGUpdater;
  BumpPtrAllocator Allocator;

  // InformationCache with CGSCC = nullptr (module-like behavior)
  InformationCache InfoCache(M, AG, Allocator, /*CGSCC*/ nullptr);

  DenseSet<const char *> Allowed({&AAUniformWorkGroupSize::ID});

  AttributorConfig AC(CGUpdater);
  AC.IsModulePass = false; // Still CGSCC context
  AC.Allowed = &Allowed;
  AC.DefaultInitializeLiveInternals = false;

  // Attributor sees ALL module functions
  Attributor A(Functions, InfoCache, AC);

  // Seed AAUniformWorkGroupSize for functions in the SCC
  for (LazyCallGraph::Node &N : C) {
    Function *Fn = &N.getFunction();
    A.getOrCreateAAFor<AAUniformWorkGroupSize>(IRPosition::function(*Fn));
  }

  LLVM_DEBUG(dbgs() << "[AMDGPUSimpleAttributor] Running Attributor...\n");

  ChangeStatus Changed = A.run();

  LLVM_DEBUG(
      dbgs() << "[AMDGPUSimpleAttributor] Attributor returned "
             << (Changed == ChangeStatus::CHANGED ? "CHANGED" : "UNCHANGED")
             << "\n");

  if (Changed == ChangeStatus::CHANGED) {
    PreservedAnalyses PA;
    PA.preserve<FunctionAnalysisManagerCGSCCProxy>();
    return PA;
  }
  return PreservedAnalyses::all();
}
