#! /usr/bin/env perl
#-*-Perl-*- This line forces emacs to use Perl mode.

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

my @nbops = qw(equals not_equals less_than greater_than not_less_than not_greater_than);
my @bbops = qw(and or);
my @ubops = qw(not);

# decls.
for my $node (@nbops, @bbops, @ubops) {
  my $n2 = "yc_${node}_node";
  print
    "    class $n2;\n".
    "    /// Shared pointer to \\ref $n2\n".
    "    typedef std::shared_ptr<$n2> ${n2}_ptr;\n\n";
}

# swig decls.
for my $node (@nbops, @bbops, @ubops) {
  my $n2 = "yc_${node}_node";
  print "\%shared_ptr(yask::$n2)\n";
}

# binary ops.
for my $node (@bbops) {
  my $n2 = "yc_${node}_node";

  print <<"END";
        /// Create a boolean $node node.
        /**
           \@returns Pointer to new \\ref $n2 object.
        */
        virtual ${n2}_ptr
        new_${node}_node(yc_bool_node_ptr lhs /**< [in] Expression before '?' sign. */,
                        yc_bool_node_ptr rhs /**< [in] Expression after '?' sign. */ );
END
}

# comparison ops.
for my $node (@nbops) {
  my $n2 = "yc_${node}_node";

  print <<"END";

        /// Create a numerical-comparison '$node' node.
        /**
           \@returns Pointer to new \\ref $n2 object.
        */
        virtual ${n2}_ptr
        new_${node}_node(yc_number_node_ptr lhs /**< [in] Expression before '?' sign. */,
                        yc_number_node_ptr rhs /**< [in] Expression after '?' sign. */ );
END
}

# binary ops.
for my $node (@bbops) {
  my $n2 = "yc_${node}_node";

  print <<"END";

    /// A boolean '$node' operator.
    /** Example: used to implement `a ?? b`.
        Created via yc_node_factory::new_${node}_node().
     */
    class $n2 : public virtual yc_bool_node {
    public:

        /// Get the left-hand-side operand.
        /** \@returns Expression node on left-hand-side of '?' sign. */
        virtual yc_bool_node_ptr
        get_lhs() =0;

        /// Get the right-hand-size operand.
        /** \@returns Expression node on right-hand-side of '?' sign. */
        virtual yc_bool_node_ptr
        get_rhs() =0;
    };
END
}

# comparison ops.
for my $node (@nbops) {
  my $n2 = "yc_${node}_node";

  print <<"END";

    /// A numerical-comparison '$node' operator.
    /** Example: used to implement `a ?? b`.
        Created via yc_node_factory::new_${node}_node().
     */
    class $n2 : public virtual yc_bool_node {
    public:

        /// Get the left-hand-side operand.
        /** \@returns Expression node on left-hand-side of '?' sign. */
        virtual yc_bool_node_ptr
        get_lhs() =0;

        /// Get the right-hand-size operand.
        /** \@returns Expression node on right-hand-side of '?' sign. */
        virtual yc_bool_node_ptr
        get_rhs() =0;
    };
END
}

# binary ops.
for my $node (@bbops) {
  my $n2 = "yc_${node}_node";
  my $n3 = $node;
  $n3 =~ s/(\w+)/\u\L$1/g;
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

# comp. ops.
for my $node (@nbops) {
  my $n2 = "yc_${node}_node";
  my $n3 = $node;
  $n3 =~ s/(\w+)/\u\L$1/g;
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
