#! /usr/bin/env perl
#-*-Perl-*- This line forces emacs to use Perl mode.

# This utility is used to generate some API code for the compiler.
# The code generated may require additional editing, so it is only
# used for one-time generation.

use strict;
use File::Basename;
use File::Path;
use lib dirname($0)."/lib";
use lib dirname($0)."/../lib";

use File::Which;
use Text::ParseWords;
use FileHandle;
use CmdLine;

$| = 1;                         # autoflush.

# num to num.
my %nops = ("add" => "+",
            "subtract" => "-",
            "multiply" => "*",
            "divide" => "/");
my %bnops = ("negate" => "-");

# num to bool.
my %nbops = ("equals" => "==",
             "not_equals" => "!=",
             "less_than" => "<",
             "greater_than" => ">",
             "not_less_than" => ">=",
             "not_greater_than" => "<=");

# bool to bool.
my %bbops = ("and" => "&&",
             "or" => "||");
my %ubops = ("not" => "!");

# decls.
for my $node (sort keys %nbops,
              sort keys %bbops,
              sort keys %ubops) {
  my $n2 = "yc_${node}_node";
  print
    "    class $n2;\n".
    "    /// Shared pointer to \\ref $n2\n".
    "    typedef std::shared_ptr<$n2> ${n2}_ptr;\n\n";
}

# swig decls.
for my $node (sort keys %nbops,
              sort keys %bbops,
              sort keys %ubops) {
  my $n2 = "yc_${node}_node";
  print "\%shared_ptr(yask::$n2)\n";
}

# binary ops.
for my $node (sort keys %bbops) {
    my $n2 = "yc_${node}_node";
    my $oper = $bbops{$node};

  print <<"END";
        /// Create a boolean $node node.
        /**
           \@returns Pointer to new \\ref $n2 object.
        */
        virtual ${n2}_ptr
        new_${node}_node(yc_bool_node_ptr lhs /**< [in] Expression before '$oper' sign. */,
                        yc_bool_node_ptr rhs /**< [in] Expression after '$oper' sign. */ );
END
}

# comparison ops.
for my $node (sort keys %nbops) {
    my $n2 = "yc_${node}_node";
    my $oper = $nbops{$node};

  print <<"END";

        /// Create a numerical-comparison '$node' node.
        /**
           \@returns Pointer to new \\ref $n2 object.
        */
        virtual ${n2}_ptr
        new_${node}_node(yc_number_node_ptr lhs /**< [in] Expression before '$oper' sign. */,
                        yc_number_node_ptr rhs /**< [in] Expression after '$oper' sign. */ );
END
}

# binary ops.
for my $node (sort keys %bbops) {
    my $n2 = "yc_${node}_node";
    my $oper = $bbops{$node};

  print <<"END";

    /// A boolean '$node' operator.
    /** Example: used to implement `a $oper b`.
        Created via yc_node_factory::new_${node}_node().
     */
    class $n2 : public virtual yc_bool_node {
    public:

        /// Get the left-hand-side operand.
        /** \@returns Expression node on left-hand-side of '$oper' sign. */
        virtual yc_bool_node_ptr
        get_lhs() =0;

        /// Get the right-hand-size operand.
        /** \@returns Expression node on right-hand-side of '$oper' sign. */
        virtual yc_bool_node_ptr
        get_rhs() =0;
    };
END
}

# comparison ops.
for my $node (sort keys %nbops) {
    my $n2 = "yc_${node}_node";
    my $oper = $nbops{$node};

  print <<"END";

    /// A numerical-comparison '$node' operator.
    /** Example: used to implement `a $oper b`.
        Created via yc_node_factory::new_${node}_node().
     */
    class $n2 : public virtual yc_bool_node {
    public:

        /// Get the left-hand-side operand.
        /** \@returns Expression node on left-hand-side of '$oper' sign. */
        virtual yc_bool_node_ptr
        get_lhs() =0;

        /// Get the right-hand-size operand.
        /** \@returns Expression node on right-hand-side of '$oper' sign. */
        virtual yc_bool_node_ptr
        get_rhs() =0;
    };
END
}

# binary ops.
for my $node (sort keys %bbops) {
    my $n2 = "yc_${node}_node";
    my $oper = $bbops{$node};
    my $n3 = $node;
    $n3 =~ s/([a-z]+)/\u\L$1/g;
    $n3 =~ s/_//g;
    $n3 .= 'Expr';
  
  print <<"END";
    ${n2}_ptr
    yc_node_factory::new_${node}_node(yc_bool_node_ptr lhs,
                                      yc_bool_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<BoolExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<BoolExpr>(rhs);
        assert(rp);
        return make_shared<$n3>(lp, rp);
    }

END
}

# comparison ops.
for my $node (sort keys %nbops) {
    my $n2 = "yc_${node}_node";
    my $oper = $nbops{$node};
    my $n3 = $node;
    $n3 =~ s/([a-z]+)/\u\L$1/g;
    $n3 =~ s/_//g;
    $n3 .= 'Expr';
  
  print <<"END";
    ${n2}_ptr
    yc_node_factory::new_${node}_node(yc_number_node_ptr lhs,
                                      yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<$n3>(lp, rp);
    }
END
}

# binary num ops.
for my $node (sort keys %nops) {
    my $n2 = "yc_${node}_node";
    my $n2p = $n2.'_ptr';
    my $oper = $nops{$node};
    print "$n2p operator$oper(yc_number_node_ptr lhs, yc_number_node_ptr rhs);\n";
    print "$n2p operator$oper(double lhs, yc_number_node_ptr rhs);\n";
    print "$n2p operator$oper(yc_number_node_ptr lhs, double);\n";
}

# binary num ops.
for my $node (sort keys %nops) {
    my $n2 = "yc_${node}_node";
    my $n2p = $n2.'_ptr';
    my $oper = $nops{$node};
    my $n3 = $node;
    $n3 =~ s/([a-z]+)/\u\L$1/g;
    $n3 =~ s/_//g;
    $n3 .= 'Expr';

    print <<"END";
    $n2p operator$oper(yc_number_node_ptr lhs, yc_number_node_ptr rhs) {
        auto lp = dynamic_pointer_cast<NumExpr>(lhs);
        assert(lp);
        auto rp = dynamic_pointer_cast<NumExpr>(rhs);
        assert(rp);
        return make_shared<$n3>(lp, rp);
    }
    $n2p operator$oper(double lhs, yc_number_node_ptr rhs) {
        return operator$oper(constNum(lhs), rhs);
    }
    $n2p operator$oper(yc_number_node_ptr lhs, double rhs) {
        return operator$oper(lhs, constNum(rhs));
    }
END
}

# binary num ops.
for my $node (sort keys %nops) {
    my $n2 = "yc_${node}_node";
    my $n2p = $n2.'_ptr';
    my $oper = $nops{$node};
    my $n3 = $node;
    $n3 =~ s/([a-z]+)/\u\L$1/g;
    $n3 =~ s/_//g;
    $n3 .= 'Expr';

    print <<"END";
%extend yask::yc_number_node {
    yask::yc_number_node_ptr __${node}__(yask::yc_number_node* rhs) {
        auto lp = \$self->clone_ast();
        auto rp = rhs->clone_ast();
        return yask::operator$oper(lp, rp);
    }
 };
END
}
