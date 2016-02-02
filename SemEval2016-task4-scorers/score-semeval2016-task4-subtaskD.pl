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
	#michael jordan   .35  .65
	die "Wrong format: ", $_ if (!/^([^\t]+)\t(0?\.\d+)\t(0?\.\d+)/);
	my ($topic, $pos, $neg) = ($1, $2, $3);
	die "Probability not in range $pos" if ($pos < -0.0001 || $pos > 1.0001);  
	die "Probability not in range $neg" if ($neg < -0.0001 || $neg > 1.0001);  

	die "Probabilities do not sum to 1: ", ($pos + $neg) if (($pos + $neg) != 1.0);

	### 1.2	. Check the prediction file format (same as input)
	$_ = <INPUT>;
	die "Wrong format: ", $_ if (!/^([^\t]+)\t(0?\.\d+)\t(0?\.\d+)/);
	my ($proposedTopic, $proposedPos, $proposedNeg) = ($1, $2, $3);
	die "Probability not in range $proposedPos" if ($proposedPos < -0.0001 || $proposedPos > 1.0001);  
	die "Probability not in range $proposedNeg" if ($proposedNeg < -0.0001 || $proposedNeg > 1.0001);  
	die "Probabilities do not sum to 1" if (($proposedPos + $proposedNeg) != 1.0);

        die "Topic mismatch!" if ($topic ne $proposedTopic);
	### 1.3. Update the statistics
	$trueStats{$topic}{'positive'} = $pos;
	$trueStats{$topic}{'negative'} = $neg;
	$proposedStats{$topic}{'positive'} = $proposedPos;
	$proposedStats{$topic}{'negative'} = $proposedNeg;
}

close(INPUT) or die;
close(GOLD) or die;

print "Format OK\n";

my $epsilon = 1 / 10000;
### 2. Initialize zero counts, compute probabilities and smooth
foreach my $topic (keys %trueStats) {
    foreach my $class (keys $trueStats{$topic}) {
	# Smooth the probabilities by epsilon
	$trueStats{$topic}{$class} = ($trueStats{$topic}{$class} + $epsilon) / (1.0 + $epsilon * (scalar keys $trueStats{$topic}));
	# Smooth the probabilities by epsilon
	$proposedStats{$topic}{$class} = ($proposedStats{$topic}{$class} + $epsilon) / (1.0 + $epsilon * (scalar keys $trueStats{$topic}));
    }
}



### 3. Calculate KL divergence for each topic and average
print "$INPUT_FILE\t";
open OUTPUT, '>:encoding(UTF-8)', $OUTPUT_FILE or die;

my $overall = 0.0;
my $numTopics = 0;
foreach my $topic (keys %trueStats) {
    my $kl = 0.0;
    foreach my $class (keys $trueStats{$topic}) {
	$kl += $trueStats{$topic}{$class} *
	    log($trueStats{$topic}{$class}/$proposedStats{$topic}{$class});
    }
    $overall += $kl;
    $numTopics ++;
    printf OUTPUT "\t%18s: KL=%0.2f\n", $topic, $kl;
}
$overall /= $numTopics;
printf OUTPUT "\tOVERALL KL divergence : %0.2f\n", $overall;
printf "%0.2f\t", $overall;

print "\n";
close(OUTPUT) or die;
