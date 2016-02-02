#!/usr/bin/perl
#
#  Author: Veselin Stoyanov
#  
#  Description: Scores SemEval-2016 task 4, subtask D
#
#  Last modified: Nov. 9, 2015
#
#

use warnings;
use strict;
use utf8;
binmode (STDIN,  ":utf8");
binmode (STDOUT, ":utf8");

my $GOLD_FILE          =  $ARGV[0];
my $INPUT_FILE         =  $ARGV[1];
my $OUTPUT_FILE        =  $INPUT_FILE . '.scored';


########################
###   MAIN PROGRAM   ###
########################

my %trueStats = ();
my %proposedStats = ();

### 1. Read the files and get the statsitics
open INPUT, '<:encoding(UTF-8)', $INPUT_FILE or die;
open GOLD,  '<:encoding(UTF-8)', $GOLD_FILE or die;

my $totalExamples = 0;
for (; <GOLD>; ) {
	s/^[ \t]+//;
	s/[ \t\n\r]+$//;

	### 1.1. Check the input file format
	#michael jordan   .3  .6  .0  .05 .05
	die "Wrong format: ", $_ if (!/^([^\t]+)\t(0?\.\d+)\t(0?\.\d+)\t(0?\.\d+)\t(0?\.\d+)\t(0?\.\d+)/);
	my $topic = $1;
	$trueStats{$topic}{'-2'} = $2;
	$trueStats{$topic}{'-1'} = $3;
	$trueStats{$topic}{'0'} = $4;
	$trueStats{$topic}{'1'} = $5;
	$trueStats{$topic}{'2'} = $6;

	my $sum = 0.0;
	foreach my $class (keys $trueStats{$topic}) {
	    my $p = $trueStats{$topic}{$class};
	    die "Number not in range $p" if ($p < -0.0001 || $p > 1.0001);  
	    $sum += $trueStats{$topic}{$class};
	}
	die "Probabilities do not sum to 1, ($sum) topic ", $topic if (abs($sum - 1.0) > .001);
	### 1.2	. Check the prediction file format (same as above)
	$_ = <INPUT>;
	die "Wrong format: ", $_ if (!/^([^\t]+)\t(0?\.\d+)\t(0?\.\d+)\t(0?\.\d+)\t(0?\.\d+)\t(0?\.\d+)/);
	my $proposedTopic = $1;
	$proposedStats{$proposedTopic}{'-2'} = $2;
	$proposedStats{$proposedTopic}{'-1'} = $3;
	$proposedStats{$proposedTopic}{'0'} = $4;
	$proposedStats{$proposedTopic}{'1'} = $5;
	$proposedStats{$proposedTopic}{'2'} = $6;

	$sum = 0.0;
	foreach my $class (keys $proposedStats{$proposedTopic}) {
	    my $p = $proposedStats{$topic}{$class};
	    die "Number not in range $p" if ($p < -0.0001 || $p > 1.0001);  
	    $sum += $proposedStats{$proposedTopic}{$class};
	}
	die "Probabilities do not sum to 1, ($sum) topic ", $proposedTopic if (abs($sum - 1.0) > .001);

        die "Topic mismatch!" if ($topic ne $proposedTopic);
}

close(INPUT) or die;
close(GOLD) or die;

print "Format OK\n";

my @labels = ('-2', '-1', '0', '1', '2');

### 2. Calculate KL divergence for each topic and average
print "$INPUT_FILE\t";
open OUTPUT, '>:encoding(UTF-8)', $OUTPUT_FILE or die;

my $overall = 0.0;
my $numTopics = 0;
foreach my $topic (keys %trueStats) {
    my $emd = 0.0;
    for my $ind1  (0 .. $#labels - 1) {
	for my $ind2 (0 .. $ind1) {
	    my $class = $labels[$ind2];
	    $emd += abs($trueStats{$topic}{$class} - $proposedStats{$topic}{$class});
	}
    }
    $overall += $emd;
    $numTopics ++;
    printf OUTPUT "\t%18s: EMD=%0.2f\n", $topic, $emd;
}
$overall /= $numTopics;
printf OUTPUT "\tOVERALL EMD : %0.2f\n", $overall;
printf "%0.2f\t", $overall;

print "\n";
close(OUTPUT) or die;
