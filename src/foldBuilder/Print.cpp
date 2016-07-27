/*****************************************************************************

YASK: Yet Another Stencil Kernel
Copyright (c) 2014-2016, Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

*****************************************************************************/

//////////// Implement methods for printing classes /////////////

#include "Print.hpp"
#include "CppIntrin.hpp"

// A grid or parameter read.
void PrintVisitorTopDown::visit(GridPoint* gp) {
    if (gp->isParam())
        _exprStr += _ph.readFromParam(_os, *gp);
    else
        _exprStr += _ph.readFromPoint(_os, *gp);
    _numCommon += _ph.getNumCommon(gp);
}

// A constant.
void PrintVisitorTopDown::visit(ConstExpr* ce) {
    _exprStr += _ph.addConstExpr(_os, ce->getVal());
    _numCommon += _ph.getNumCommon(ce);
}

// Some code.
void PrintVisitorTopDown::visit(CodeExpr* ce) {
    _exprStr += _ph.addCodeExpr(_os, ce->getCode());
    _numCommon += _ph.getNumCommon(ce);
}

// A generic unary operator.
void PrintVisitorTopDown::visit(UnaryExpr* ue) {
    _exprStr += ue->getOpStr();
    ue->getRhs()->accept(this);
    _numCommon += _ph.getNumCommon(ue);
}

// A generic binary operator.
void PrintVisitorTopDown::visit(BinaryExpr* be) {
    _exprStr += "(";
    be->getLhs()->accept(this); // adds LHS to _exprStr.
    _exprStr += " " + be->getOpStr() + " ";
    be->getRhs()->accept(this); // adds RHS to _exprStr.
    _exprStr += ")";
    _numCommon += _ph.getNumCommon(be);
}

// A commutative operator.
void PrintVisitorTopDown::visit(CommutativeExpr* ce) {
    _exprStr += "(";
    ExprPtrVec& ops = ce->getOps();
    int opNum = 0;
    for (auto ep : ops) {
        if (opNum > 0)
            _exprStr += " " + ce->getOpStr() + " ";
        ep->accept(this);   // adds operand to _exprStr;
        opNum++;
    }
    _exprStr += ")";
    _numCommon += _ph.getNumCommon(ce);
}

// An equals operator.
void PrintVisitorTopDown::visit(EqualsExpr* ee) {

    // Get RHS and clear expr.
    ee->getRhs()->accept(this); // writes to _exprStr;
    string rhs = getExprStrAndClear();

    // Write statement with embedded rhs.
    GridPointPtr gpp = ee->getLhs();
    _os << _ph.getLinePrefix() << _ph.writeToPoint(_os, *gpp, rhs) << _ph.getLineSuffix();

    // note: _exprStr is now empty.
    // note: no need to update num-common.
}


// If 'comment' is set, use it for the comment.
// Return stream to continue w/RHS.
ostream& PrintVisitorBottomUp::makeNextTempVar(Expr* ex, string comment) {
    _exprStr = _ph.makeVarName();
    if (ex) {
        _tempVars[ex] = _exprStr;
        if (comment.length() == 0)
            _os << endl << " // " << _exprStr << " = " << ex->makeStr() << "." << endl;
    }
    if (comment.length())
        _os << endl << " // " << _exprStr << " = " << comment << "." << endl;
    _os << _ph.getLinePrefix() << _ph.getVarType() << " " << _exprStr << " = ";
    return _os;
}

// Look for existing var.
// Then, use top-down method for simple exprs.
// Return true if successful.
// FIXME: this causes all nodes in an expr above a certain
// point to fail top-down because it looks at the original expr,
// not the one with temp vars.
bool PrintVisitorBottomUp::tryTopDown(Expr* ex, bool leaf) {

    // First, determine whether this expr has already been evaluated.
    auto p = _tempVars.find(ex);
    if (p != _tempVars.end()) {

        // if so, just use the existing var.
        _exprStr = p->second;
        return true;
    }
        
    // Use top down if leaf node or <= maxPoints points in ex.
    if (leaf || ex->getNumNodes() <= _maxPoints) {

        // use a top-down printer to render the expr.
        PrintVisitorTopDown* topDown = newPrintVisitorTopDown();
        ex->accept(topDown);

        // were there any common subexprs found?
        bool ok = topDown->getNumCommon() == 0;

        // if no common subexprs, use the rendered expression.
        if (ok)
            _exprStr = topDown->getExprStr();
            
        // if a leaf node is common, make a var for it.
        else if (leaf) {
            makeNextTempVar(ex) << topDown->getExprStr() << _ph.getLineSuffix();
            ok = true;
        }

        delete topDown;
        return ok;
    }

    return false;
}

// A grid or param point: just set expr.
void PrintVisitorBottomUp::visit(GridPoint* gp) {
    tryTopDown(gp, true);
}

// A constant: just set expr.
void PrintVisitorBottomUp::visit(ConstExpr* ce) {
    tryTopDown(ce, true);
}

// Code: just set expr.
void PrintVisitorBottomUp::visit(CodeExpr* ce) {
    tryTopDown(ce, true);
}

// A unary operator.
void PrintVisitorBottomUp::visit(UnaryExpr* ue) {

    // Try top-down on whole expression.
    // Example: '-a' creates no immediate output,
    // and '-a' is saved in _exprStr.
    if (tryTopDown(ue, false))
        return;

    // Expand the RHS, then apply operator to result.
    // Example: '-(a * b)' might output the following:
    // temp1 = a * b;
    // temp2 = -temp1;
    // with 'temp2' saved in _exprStr.
    ue->getRhs()->accept(this); // sets _exprStr.
    string rhs = getExprStrAndClear();
    makeNextTempVar(ue) << ue->getOpStr() << ' ' << rhs << _ph.getLineSuffix();
}

// A binary operator.
void PrintVisitorBottomUp::visit(BinaryExpr* be) {

    // Try top-down on whole expression.
    // Example: 'a/b' creates no immediate output,
    // and 'a/b' is saved in _exprStr.
    if (tryTopDown(be, false))
        return;

    // Expand both sides, then apply operator to result.
    // Example: '(a * b) / (c * d)' might output the following:
    // temp1 = a * b;
    // temp2 = b * c;
    // temp3 = temp1 / temp2;
    // with 'temp3' saved in _exprStr.
    be->getLhs()->accept(this); // sets _exprStr.
    string lhs = getExprStrAndClear();
    be->getRhs()->accept(this); // sets _exprStr.
    string rhs = getExprStrAndClear();
    makeNextTempVar(be) << lhs << ' ' << be->getOpStr() << ' ' << rhs << _ph.getLineSuffix();
}

// A commutative operator.
void PrintVisitorBottomUp::visit(CommutativeExpr* ce) {

    // Try top-down on whole expression.
    // Example: 'a*b' creates no immediate output,
    // and 'a*b' is saved in _exprStr.
    if (tryTopDown(ce, false))
        return;

    // Make separate assignment for N-1 operands.
    // Example: 'a + b + c + d' might output the following:
    // temp1 = a + b;
    // temp2 = temp1 + c;
    // temp3 = temp2 = d;
    // with 'temp3' left in _exprStr;
    ExprPtrVec& ops = ce->getOps();
    assert(ops.size() > 1);
    string lhs, exStr;
    int opNum = 0;
    for (auto ep : ops) {
        opNum++;

        // eval the operand; sets _exprStr.            
        ep->accept(this);
        string opStr = getExprStrAndClear();

        // first operand; just save as LHS for next iteration.
        if (opNum == 1) {
            lhs = opStr;
            exStr = ep->makeStr();
        }

        // subsequent operands.
        // makes separate assignment for each one.
        // result is kept as LHS of next one.
        else {

            // Use whole expression only for the last step.
            Expr* ex = (opNum == (int)ops.size()) ? ce : NULL;

            // Add RHS to partial-result comment.
            exStr += ' ' + ce->getOpStr() + ' ' + ep->makeStr();

            // Output this step.
            makeNextTempVar(ex, exStr) << lhs << ' ' << ce->getOpStr() << ' ' <<
                opStr << _ph.getLineSuffix();
            lhs = getExprStr(); // result used in next iteration, if any.
        }
    }

    // note: _exprStr contains result of last operation.
}

// An equality.
void PrintVisitorBottomUp::visit(EqualsExpr* ee) {

    // Eval RHS.
    Expr* rp = ee->getRhs().get();
    rp->accept(this); // sets _exprStr.
    string rhs = _exprStr;

    // Assign RHS to a temp var.
    makeNextTempVar(rp) << rhs << _ph.getLineSuffix(); // sets _exprStr.
    string tmp = getExprStrAndClear();

    // Write temp var to grid.
    GridPointPtr gpp = ee->getLhs();
    _os << endl << " // Save result to " << gpp->makeStr() << ":" << endl;
    _os << _ph.getLinePrefix() << _ph.writeToPoint(_os, *gpp, tmp) << _ph.getLineSuffix();

    // note: _exprStr is now empty.
}
    
// Only want to visit the RHS of an equation.
void POVRayPrintVisitor::visit(EqualsExpr* ee) {
    ee->getRhs()->accept(this);      
}
    
// A point: output it.
void POVRayPrintVisitor::visit(GridPoint* gp) {
    _numPts++;

    // Pick a color based on its distance.
    size_t ci = gp->max();
    ci %= _colors.size();
        
    _os << "point(" + _colors[ci] + ", " << gp->makeValStr() << ")" << endl;
}


// Print out a stencil in human-readable form, for debug or documentation.
void PseudoPrinter::print(ostream& os) {

    os << "Stencil '" << _stencil.getName() << "'pseudo-code:" << endl;

    // Loop through all equations.
    for (auto& eq : _equations) {
            
        os << endl << " ////// Equation '" << eq.name <<
            "' //////" << endl;

        CounterVisitor cv;
        eq.grids.acceptToAll(&cv);
        PrintHelper ph(&cv, "temp", "real", " ", ".\n");

        os << " // Top-down stencil calculation:" << endl;
        PrintVisitorTopDown pv1(os, ph);
        eq.grids.acceptToAll(&pv1);
            
        os << endl << " // Bottom-up stencil calculation:" << endl;
        PrintVisitorBottomUp pv2(os, ph, _exprSize);
        eq.grids.acceptToAll(&pv2);
    }
}

// Print out a stencil in POVRay form.
void POVRayPrinter::print(ostream& os) {

    os << "#include \"stencil.inc\"" << endl;
    int cpos = 25;
    os << "camera { location <" <<
        cpos << ", " << cpos << ", " << cpos << ">" << endl <<
        "  look_at <0, 0, 0>" << endl <<
        "}" << endl;

    // Loop through all equations.
    for (auto& eq : _equations) {

        // TODO: separate mutiple grids.
        POVRayPrintVisitor pv(os);
        eq.grids.acceptToFirst(&pv);
        os << " // " << pv.getNumPoints() << " stencil points" << endl;
    }
}

// Print YASK code.
void YASKCppPrinter::printCode(ostream& os) {

    os << "// Automatically generated code; do not edit." << endl;

    os << endl << "////// Implementation of the '" << _stencil.getName() <<
        "' stencil //////" << endl;
    os << endl << "namespace yask {" << endl;

    // Create the overall context class.
    {
        // get stats for just one element (not cluster).
        CounterVisitor cve;
        _grids.acceptToFirst(&cve);
        IntTuple maxHalos;

        os << endl << " ////// Overall stencil-context class //////" << endl <<
            "struct " << _context << " : public StencilContext {" << endl;

        // Grids.
        os << endl << " // Grids." << endl;
        map<Grid*, string> typeNames, dimArgs, padArgs;
        for (auto gp : _grids) {
            assert (!gp->isParam());
            string grid = gp->getName();

            // Type name & ctor params.
            // Name in kernel is 'Grid_' followed by dimensions.
            string typeName = "Grid_";
            string dimArg, padArg;
            for (auto dim : gp->getDims()) {
                string ucDim = allCaps(dim);
                typeName += ucDim;

                // don't want the time dimension during construction.
                // TODO: make this more generic.
                if (dim != "t") {
                    dimArg += "d" + dim + ", ";

                    // Halo for this dimension.
                    int halo = cve.getHalo(gp, dim);
                    string hvar = grid + "_halo_" + dim;
                    os << " const idx_t " << hvar << " = " << halo << ";" << endl;

                    // Update max halo.
                    int* mh = maxHalos.lookup(dim);
                    if (mh)
                        *mh = max(*mh, halo);
                    else
                        maxHalos.addDim(dim, halo);

                    // Total padding = halo + extra.
                    padArg += hvar + " + p" + dim + ", ";
                }
            }
            typeNames[gp] = typeName;
            dimArgs[gp] = dimArg;
            padArgs[gp] = padArg;
            os << " " << typeName << "* " << grid << "; // ";
            if (_equations.getEqGrids().count(gp) == 0)
                os << "not ";
            os << "updated by stencil." << endl;
        }

        // Max halos.
        os << endl << " // Max halos across all grids." << endl;
        for (auto dim : maxHalos.getDims())
            os << " const idx_t max_halo_" << dim << " = " <<
                maxHalos.getVal(dim) << ";" << endl;

        // Parameters.
        map<Param*, string> paramTypeNames, paramDimArgs;
        os << endl << " // Parameters." << endl;
        for (auto pp : _params) {
            assert(pp->isParam());
            string param = pp->getName();

            // Type name.
            // Name in kernel is 'GenericGridNd<real_t>'.
            ostringstream oss;
            oss << "GenericGrid" << pp->size() << "d<real_t";
            if (pp->size()) {
                oss << ",Layout_";
                for (int dn = pp->size(); dn > 0; dn--)
                    oss << dn;
            }
            oss << ">";
            string typeName = oss.str();

            // Ctor params.
            string dimArg = pp->makeValStr();
            
            paramTypeNames[pp] = typeName;
            paramDimArgs[pp] = dimArg;
            os << " " << typeName << "* " << param << ";" << endl;
        }

        // Ctor.
        os << endl << " " << _context << "() {" << endl <<
            "  name = \"" << _stencil.getName() << "\";" << endl;

        // Init grid ptrs.
        for (auto gp : _grids) {
            string grid = gp->getName();
            os << "  " << grid << " = 0;" << endl;
        }

        // Init param ptrs.
        for (auto pp : _params) {
            string param = pp->getName();
            os << "  " << param << " = 0;" << endl;
        }

        // end of ctor.
        os << " }" << endl;

        // Allocate grids.
        os << endl << " virtual void allocGrids() {" << endl;
        os << "  gridPtrs.clear();" << endl;
        os << "  eqGridPtrs.clear();" << endl;
        for (auto gp : _grids) {
            string grid = gp->getName();
            os << "  " << grid << " = new " << typeNames[gp] <<
                "(" << dimArgs[gp] << padArgs[gp] << "\"" << grid << "\");" << endl <<
                "  gridPtrs.push_back(" << grid << ");" << endl;

            // Grids w/equations.
            if (_equations.getEqGrids().count(gp))
                os << "  eqGridPtrs.push_back(" << grid  << ");" << endl;
        }
        os << " }" << endl;

        // Allocate params.
        os << endl << " virtual void allocParams() {" << endl;
        os << "  paramPtrs.clear();" << endl;
        for (auto pp : _params) {
            string param = pp->getName();
            os << "  " << param << " = new " << paramTypeNames[pp] <<
                "(" << paramDimArgs[pp] << ");" << endl <<
                "  paramPtrs.push_back(" << param << ");" << endl;
        }
        os << " }" << endl;

        // end of context.
        os << "};" << endl;
    }
        
    // Loop through all equations.
    for (auto& eq : _equations) {

        os << endl << " ////// Stencil equation '" << eq.name <<
            "' //////" << endl;

        os << endl << "struct Stencil_" << eq.name << " {" << endl <<
            " std::string name = \"" << eq.name << "\";" << endl;

        // Ops for this equation.
        CounterVisitor fpops;
        eq.grids.acceptToFirst(&fpops);
            
        // Example computation.
        os << endl << " // " << fpops.getNumOps() << " FP operation(s) per point:" << endl;
        addComment(os, eq.grids);
        os << " const int scalar_fp_ops = " << fpops.getNumOps() << ";" << endl;

        // Init code.
        {
            os << endl << " // All grids updated by this equation." << endl <<
                " std::vector<RealVecGridBase*> eqGridPtrs;" << endl;

            os << " void init(" << _context << "& context) {" << endl;

            // Grids w/equations.
            os << "  eqGridPtrs.clear();" << endl;
            for (auto gp : eq.grids) {
                os << "  eqGridPtrs.push_back(context." << gp->getName() << ");" << endl;
            }
            os << " }" << endl;
        }
            
        // Scalar code.
        {
            // C++ scalar print assistant.
            CounterVisitor cv;
            eq.grids.acceptToFirst(&cv);
            CppPrintHelper* sp = new CppPrintHelper(&cv, "temp", "real_t", " ", ";\n");
            
            // Stencil-calculation code.
            // Function header.
            os << endl << " // Calculate one scalar result relative to indices " <<
                _dimCounts.makeDimStr(", ") << "." << endl;
            os << " void calc_scalar(" << _context << "& context, " <<
                _dimCounts.makeDimStr(", ", "idx_t ") << ") {" << endl;

            // C++ code generator.
            // The visitor is accepted at all nodes in the AST;
            // for each node in the AST, code is generated and
            // stored in the expression-string in the visitor.
            // Visit only first expression in each, since we don't want clustering.
            PrintVisitorBottomUp pcv(os, *sp, _exprSize);
            eq.grids.acceptToFirst(&pcv);

            // End of function.
            os << "} // scalar calculation." << endl;

            delete sp;
        }

        // Cluster/Vector code.
            
        // Create vectors needed to implement.
        // The visitor is accepted at all nodes in the AST;
        // for each grid access node in the AST, the vectors
        // needed are determined and saved in the visitor.
        {
            // Create vector info for this equation.
            VecInfoVisitor vv(_foldLengths);
            eq.grids.acceptToAll(&vv);

#if 1
            // Reorder based on vector info.
            ExprReorderVisitor erv(vv);
            eq.grids.acceptToAll(&erv);
#endif
            
            // C++ vector print assistant.
            CounterVisitor cv;
            eq.grids.acceptToFirst(&cv);
            CppVecPrintHelper* vp = newPrintHelper(vv, cv);
            
            // Stencil-calculation code.
            // Function header.
            int numResults = _foldLengths.product() * _clusterLengths.product();
            os << endl << " // Calculate " << numResults <<
                " result(s) relative to indices " << _dimCounts.makeDimStr(", ") <<
                " in a '" << _clusterLengths.makeDimValStr(" * ") << "' cluster of '" <<
                _foldLengths.makeDimValStr(" * ") << "' vector(s)." << endl;
            os << " // Indices must be normalized, i.e., already divided by VLEN_*." << endl;
            os << " // SIMD calculations use " << vv.getNumPoints() <<
                " vector block(s) created from " << vv.getNumAlignedVecs() <<
                " aligned vector-block(s)." << endl;
            os << " // There are " << (fpops.getNumOps() * numResults) <<
                " FP operation(s) per cluster." << endl;

            os << " void calc_cluster(" << _context << "& context, " <<
                _dimCounts.makeDimStr(", ", "idx_t ", "v") << ") {" << endl;

            // Element indices.
            os << endl << " // Un-normalized indices." << endl;
            for (auto dim : _dimCounts.getDims()) {
                auto p = _foldLengths.lookup(dim);
                os << " idx_t " << dim << " = " << dim << "v";
                if (p) os << " * " << *p;
                os << ";" << endl;
            }
                
            // Code generator visitor.
            // The visitor is accepted at all nodes in the AST;
            // for each node in the AST, code is generated and
            // stored in the expression-string in the visitor.
            PrintVisitorBottomUp pcv(os, *vp, _exprSize);
            eq.grids.acceptToAll(&pcv);

            // End of function.
            os << "} // vector calculation." << endl;

            // Generate prefetch code for no specific direction and then each
            // orthogonal direction.
            for (int diri = -1; diri < _dimCounts.size(); diri++) {

                // Create a direction object.
                // If diri < 0, there is no direction.
                // If diri >= 0, add a direction.
                IntTuple dir;
                if (diri >= 0) {
                    string dim = _dimCounts.getDims()[diri];
                    dir.addDim(dim, 1);

                    // Don't need prefetch in time dimension.
                    if (dim == "t")
                        continue;
                }

                // Function header.
                os << endl << " // Prefetches cache line(s) ";
                if (dir.size())
                    os << "for leading edge of stencil in '+" <<
                        dir.getDirName() << "' direction ";
                else
                    os << "for entire stencil ";
                os << " relative to indices " << _dimCounts.makeDimStr(", ") <<
                    " in a '" << _clusterLengths.makeDimValStr(" * ") << "' cluster of '" <<
                    _foldLengths.makeDimValStr(" * ") << "' vector(s)." << endl;
                os << " // Indices must be normalized, i.e., already divided by VLEN_*." << endl;

                os << " template<int level> void prefetch_cluster";
                if (dir.size())
                    os << "_" << dir.getDirName();
                os << "(" << _context << "& context, " <<
                    _dimCounts.makeDimStr(", ", "idx_t ", "v") << ") {" << endl;

                // C++ prefetch code.
                vp->printPrefetches(os, dir);

                // End of function.
                os << "}" << endl;

            } // direction.

            delete vp;
        }

        os << "};" << endl; // end of class.
            
    } // stencil equations.

    // Create a class for all equations.
    os << endl << " ////// Overall stencil-equations class //////" << endl <<
        "template <typename ContextClass>" << endl <<
        "struct StencilEquations_" << _stencil.getName() <<
        " : public StencilEquations {" << endl;

    // Stencil equation objects.
    os << endl << " // Stencils." << endl;
    for (auto& eq : _equations)
        os << " StencilTemplate<Stencil_" << eq.name << "," <<
            _context << "> stencil_" << eq.name << ";" << endl;

    // Ctor.
    os << endl << " StencilEquations_" << _stencil.getName() << "() {" << endl <<
        "name = \"" << _stencil.getName() << "\";" << endl;

    // Push stencils to list.
    for (auto& eq : _equations)
        os << "  stencils.push_back(&stencil_" << eq.name << ");" << endl;
    os << " }" << endl;

    os << "};" << endl;
    os << "} // namespace yask." << endl;
        
}

// Print YASK macros.
// TODO: many hacks below assume certain dimensions and usage model
// by the kernel. Need to improve kernel to make it more flexible
// and then communicate info more generically.
void YASKCppPrinter::printMacros(ostream& os) {
    os << "// Automatically generated code; do not edit." << endl;

    os << endl;
    os << "// Stencil:" << endl;
    os << "#define STENCIL_NAME \"" << _stencil.getName() << "\"" << endl;
    os << "#define STENCIL_IS_" << allCaps(_stencil.getName()) << " (1)" << endl;
    os << "#define STENCIL_CONTEXT " << _context << endl;
    os << "#define STENCIL_EQUATIONS StencilEquations_" << _stencil.getName() <<
        "<" << _context << ">" << endl;

    os << endl;
    os << "// Dimensions:" << endl;
    for (auto dim : _dimCounts.getDims()) {
        os << "#define USING_DIM_" << allCaps(dim) << " (1)" << endl;
    }
        
    // Vec/cluster lengths.
    os << endl;
    os << "// One vector fold: " << _foldLengths.makeDimValStr(" * ") << endl;
    for (auto dim : _foldLengths.getDims()) {
        string ucDim = allCaps(dim);
        os << "#define VLEN_" << ucDim << " (" << _foldLengths.getVal(dim) << ")" << endl;
    }
    os << "#define VLEN (" << _foldLengths.product() << ")" << endl;
    os << "#define VLEN_FIRST_DIM_IS_UNIT_STRIDE (" <<
        (IntTuple::getDefaultFirstInner() ? 1 : 0) << ")" << endl;
    os << "#define USING_UNALIGNED_LOADS (" <<
        (_allowUnalignedLoads ? 1 : 0) << ")" << endl;

    os << endl;
    os << "// Cluster of vector folds: " << _clusterLengths.makeDimValStr(" * ") << endl;
    for (auto dim : _clusterLengths.getDims()) {
        string ucDim = allCaps(dim);
        os << "#define CLEN_" << ucDim << " (" << _clusterLengths.getVal(dim) << ")" << endl;
    }
    os << "#define CLEN (" << _clusterLengths.product() << ")" << endl;
}
